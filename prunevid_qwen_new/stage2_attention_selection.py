"""
Stage 2: 基于注意力的Token选择
===========================

实现PruneVid的第二阶段：利用LLM的注意力机制选择与问题最相关的视觉token。

核心思想（参考论文图1(c)）：
LLM在推理过程中会自然地关注与问题相关的视觉区域。通过提取问题token到
视觉token的交叉注意力权重，我们可以识别哪些视觉token对回答问题最重要。

论文方法（第3.3节）：
1. 在LLM的第M层提取attention weights
2. 提取问题→视觉的交叉注意力子矩阵 A_qv
3. 计算每个视觉token的重要性：max pooling over 问题tokens和attention heads
4. 选择top α%的视觉token
5. 更新序列和KV cache配置

论文公式(5)：
A_qv^(M) = A^(M)[N_q:, :N'_v]

其中N_q是问题token数，N'_v是视觉token数（Stage 1合并后）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from config import PruneVidConfig
from utils import compute_compression_stats


class AttentionBasedTokenSelector:
    """
    基于注意力的Token选择器

    通过LLM内部的注意力机制识别和保留最重要的视觉token。
    """

    def __init__(self, config: PruneVidConfig):
        """
        初始化

        Args:
            config: PruneVid配置对象
        """
        self.config = config
        self.keep_ratio = config.keep_ratio
        self.pruning_layer = config.pruning_layer
        self.aggregation = config.attention_aggregation
        self.verbose = config.verbose

        # 存储hook相关信息
        self.hook_handle = None
        self.attention_weights = None
        self.visual_token_start = None
        self.visual_token_end = None
        self.selected_indices = None

        # 统计信息
        self.last_stats = {}

    def setup_hook(
        self,
        target_layer: nn.Module,
        visual_token_start: int,
        visual_token_end: int
    ):
        """
        在指定层注册forward hook以提取注意力权重

        Args:
            target_layer: 要监控的transformer层（通常是model.layers[M]）
            visual_token_start: 视觉token起始位置（Stage 1合并后的新位置）
            visual_token_end: 视觉token结束位置
        """
        self.visual_token_start = visual_token_start
        self.visual_token_end = visual_token_end
        self.attention_weights = None
        self.selected_indices = None

        # 定义hook函数
        def attention_hook(module, input, output):
            """
            Hook函数：提取attention weights

            Qwen2.5-VL的attention层输出格式：
            output = (hidden_states, attention_weights, ...)
            attention_weights: [batch, num_heads, seq_len, seq_len]
            """
            if len(output) > 1 and output[1] is not None:
                # output[1]是attention_weights
                self.attention_weights = output[1].detach()
                if self.verbose:
                    print(f"[Stage 2] 捕获attention weights, shape: {self.attention_weights.shape}")

        # 移除旧的hook（如果有）
        if self.hook_handle is not None:
            self.hook_handle.remove()

        # 注册新hook
        self.hook_handle = target_layer.register_forward_hook(attention_hook)

        if self.verbose:
            print(f"[Stage 2] 在层 {self.pruning_layer} 注册attention hook")
            print(f"[Stage 2] 视觉token范围: [{visual_token_start}, {visual_token_end})")

    def select_tokens(self) -> torch.Tensor:
        """
        基于捕获的attention weights选择重要的视觉token

        Returns:
            selected_indices: [num_selected] 选中的视觉token的全局索引
        """
        if self.attention_weights is None:
            raise RuntimeError("没有捕获到attention weights！请先调用setup_hook并运行forward")

        if self.visual_token_start is None or self.visual_token_end is None:
            raise RuntimeError("视觉token位置未设置！")

        # attention_weights: [batch, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = self.attention_weights.shape

        # 假设batch_size=1
        if batch_size != 1:
            raise NotImplementedError("当前实现只支持batch_size=1")

        attn = self.attention_weights[0]  # [num_heads, seq_len, seq_len]

        # 提取交叉注意力：文本token → 视觉token
        # 文本token在视觉token之前和之后
        # 假设问题token在视觉token之前
        text_token_end = self.visual_token_start
        text_token_start = 0  # 简化假设：问题从开头开始

        if text_token_end <= text_token_start:
            raise RuntimeError(f"无效的文本token范围: [{text_token_start}, {text_token_end})")

        # 提取文本→视觉的注意力子矩阵
        # [num_heads, num_text_tokens, num_visual_tokens]
        cross_attention = attn[:, text_token_start:text_token_end, self.visual_token_start:self.visual_token_end]

        if self.verbose:
            print(f"[Stage 2] 交叉注意力 shape: {cross_attention.shape}")
            print(f"  文本tokens: {text_token_end - text_token_start}")
            print(f"  视觉tokens: {self.visual_token_end - self.visual_token_start}")

        # 计算每个视觉token的重要性分数
        # 策略：max-max（论文使用）或mean
        num_visual = cross_attention.shape[2]

        if self.aggregation == "max":
            # Max over 文本token维度 -> [num_heads, num_visual]
            max_over_text = cross_attention.max(dim=1)[0]
            # Max over attention heads -> [num_visual]
            importance_scores = max_over_text.max(dim=0)[0]
        elif self.aggregation == "mean":
            # Mean over 文本token维度 -> [num_heads, num_visual]
            mean_over_text = cross_attention.mean(dim=1)
            # Mean over attention heads -> [num_visual]
            importance_scores = mean_over_text.mean(dim=0)
        else:
            raise ValueError(f"不支持的聚合方式: {self.aggregation}")

        if self.verbose:
            print(f"[Stage 2] 重要性分数统计:")
            print(f"  Min: {importance_scores.min().item():.6f}")
            print(f"  Max: {importance_scores.max().item():.6f}")
            print(f"  Mean: {importance_scores.mean().item():.6f}")

        # 选择top-k
        num_to_keep = max(1, int(num_visual * self.keep_ratio))
        top_k_values, top_k_indices = torch.topk(importance_scores, num_to_keep, largest=True)

        # 转换为全局索引
        global_indices = top_k_indices + self.visual_token_start

        # 排序（保持原始顺序）
        global_indices_sorted, _ = torch.sort(global_indices)

        self.selected_indices = global_indices_sorted

        if self.verbose:
            print(f"[Stage 2] 选择 top {num_to_keep}/{num_visual} 视觉tokens ({self.keep_ratio*100:.1f}%)")
            print(f"  选中索引范围: [{global_indices_sorted[0].item()}, {global_indices_sorted[-1].item()}]")

        # 统计
        stats = compute_compression_stats(num_visual, num_to_keep)
        self.last_stats = stats

        return global_indices_sorted

    def apply_selection(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        应用token选择，更新序列

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [3, batch, seq_len]
            position_embeddings: Tuple of (cos, sin)
            attention_mask: Optional [batch, 1, seq_len, seq_len]

        Returns:
            pruned_hidden_states: [batch, new_seq_len, hidden_dim]
            pruned_position_ids: [3, batch, new_seq_len]
            pruned_position_embeddings: Tuple of (cos, sin)
            pruned_attention_mask: Optional [batch, 1, new_seq_len, new_seq_len]
        """
        if self.selected_indices is None:
            raise RuntimeError("尚未选择token！请先调用select_tokens")

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 构建保留的索引：文本token（全部保留） + 选中的视觉token
        # [0, ..., visual_start-1] + [selected visual] + [visual_end, ..., seq_len-1]

        text_before_indices = torch.arange(0, self.visual_token_start, device=hidden_states.device)
        text_after_indices = torch.arange(self.visual_token_end, seq_len, device=hidden_states.device)

        # 合并所有保留的索引
        all_kept_indices = torch.cat([
            text_before_indices,
            self.selected_indices,
            text_after_indices
        ], dim=0)

        # 应用索引选择
        pruned_hidden = hidden_states[:, all_kept_indices, :]
        pruned_pos_ids = position_ids[:, :, all_kept_indices]

        cos, sin = position_embeddings
        pruned_cos = cos[:, :, all_kept_indices, :]
        pruned_sin = sin[:, :, all_kept_indices, :]
        pruned_pos_emb = (pruned_cos, pruned_sin)

        # 更新attention mask（如果有）
        if attention_mask is not None:
            # attention_mask: [batch, 1, seq_len, seq_len] (causal mask)
            # 需要选择对应的行和列
            pruned_mask = attention_mask[:, :, all_kept_indices, :]
            pruned_mask = pruned_mask[:, :, :, all_kept_indices]
        else:
            pruned_mask = None

        if self.verbose:
            new_seq_len = pruned_hidden.shape[1]
            print(f"[Stage 2] 序列长度: {seq_len} -> {new_seq_len}")
            num_visual_kept = self.selected_indices.shape[0]
            visual_end_new = self.visual_token_start + num_visual_kept
            print(f"[Stage 2] 新的视觉token范围: [{self.visual_token_start}, {visual_end_new})")

        return pruned_hidden, pruned_pos_ids, pruned_pos_emb, pruned_mask

    def get_selection_info(self) -> Dict:
        """
        获取token选择的详细信息（用于Stage 3的KV cache压缩）

        Returns:
            info: 包含以下键的字典
                - pruning_layer: 剪枝层索引
                - visual_token_start: 视觉token起始位置（原始）
                - visual_token_end: 视觉token结束位置（原始）
                - selected_visual_indices: 选中的视觉token索引（相对于visual_start）
                - kept_all_indices: 所有保留的token的全局索引
        """
        if self.selected_indices is None:
            return {}

        # 选中的视觉token相对于visual_start的索引
        relative_selected = self.selected_indices - self.visual_token_start

        return {
            "pruning_layer": self.pruning_layer,
            "visual_token_start": self.visual_token_start,
            "visual_token_end": self.visual_token_end,
            "selected_visual_indices": relative_selected,
            "selected_global_indices": self.selected_indices,
        }

    def remove_hook(self):
        """移除forward hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            if self.verbose:
                print("[Stage 2] 移除attention hook")

    def reset(self):
        """重置状态"""
        self.attention_weights = None
        self.selected_indices = None
        self.last_stats = {}

    def __del__(self):
        """析构时确保移除hook"""
        self.remove_hook()


# 导出
__all__ = ["AttentionBasedTokenSelector"]
