"""
Stage 2: 基于LLM注意力的Token选择

在LLM的中间层（默认第10层）提取问题到视觉token的交叉注意力，
计算每个视觉token的重要性，选择top-k个最相关的token。

论文3.3节：
- 在第M层提取注意力权重 A^(M)
- 提取问题到视觉的交叉注意力 A^(M)_qv
- 对每个视觉token，计算最大注意力值（max over question tokens, max over heads）
- 选择top alpha%的token
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

try:
    from .utils import compute_token_importance_from_attention, select_top_k_tokens
    from .stage3_cache import PruneVidDynamicCache
except ImportError:
    from utils import compute_token_importance_from_attention, select_top_k_tokens
    from stage3_cache import PruneVidDynamicCache


class AttentionBasedTokenSelector:
    """
    基于注意力的Token选择器

    在LLM的指定层提取注意力权重，计算视觉token的重要性，
    并选择最重要的token进行保留。
    """

    def __init__(
        self,
        pruning_layer: int = 10,
        keep_ratio: float = 0.4,
        aggregation: str = "max",
        verbose: bool = False,
    ):
        """
        Args:
            pruning_layer: 在哪一层进行剪枝（0-indexed）
            keep_ratio: 保留比例（alpha in paper）
            aggregation: 注意力聚合方式（'max' or 'mean'）
            verbose: 是否打印详细信息
        """
        self.pruning_layer = pruning_layer
        self.keep_ratio = keep_ratio
        self.aggregation = aggregation
        self.verbose = verbose

        # 用于存储剪枝信息
        self.last_kept_indices = None
        self.last_importance_scores = None

    def compute_importance(
        self,
        attention_weights: torch.Tensor,
        visual_token_start: int,
        visual_token_end: int,
    ) -> torch.Tensor:
        """
        从注意力权重计算视觉token的重要性

        Args:
            attention_weights: [B, num_heads, seq_len, seq_len] 注意力权重
            visual_token_start: 视觉token起始位置
            visual_token_end: 视觉token结束位置

        Returns:
            importance: [B, num_visual_tokens] 每个视觉token的重要性
        """
        B, num_heads, seq_len, _ = attention_weights.shape
        num_visual_tokens = visual_token_end - visual_token_start

        # 提取文本到视觉的交叉注意力
        # text_to_visual: [B, num_heads, num_text_tokens, num_visual_tokens]
        text_to_visual = attention_weights[:, :, visual_token_end:, visual_token_start:visual_token_end]

        if text_to_visual.shape[2] == 0:
            # 如果没有文本token，使用全局注意力
            text_to_visual = attention_weights[:, :, :, visual_token_start:visual_token_end]

        # 计算重要性（max-max策略）
        importance = compute_token_importance_from_attention(
            text_to_visual, aggregation=self.aggregation
        )

        return importance

    def select_tokens(
        self,
        importance_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据重要性分数选择token

        Args:
            importance_scores: [num_visual_tokens] 重要性分数

        Returns:
            kept_indices: 保留的token索引
            pruned_indices: 被剪枝的token索引
        """
        kept_indices, pruned_indices = select_top_k_tokens(
            importance_scores, self.keep_ratio
        )

        # 保存用于调试
        self.last_kept_indices = kept_indices
        self.last_importance_scores = importance_scores

        return kept_indices, pruned_indices

    def __call__(
        self,
        attention_weights: torch.Tensor,
        visual_token_start: int,
        visual_token_end: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行token选择

        Args:
            attention_weights: [B, num_heads, seq_len, seq_len]
            visual_token_start: 视觉token起始位置
            visual_token_end: 视觉token结束位置

        Returns:
            kept_indices: 保留的视觉token索引（相对于visual_token_start）
            stats: 统计信息
        """
        # 计算重要性
        importance = self.compute_importance(
            attention_weights, visual_token_start, visual_token_end
        )

        # 处理batch（目前只支持batch_size=1）
        if importance.dim() == 2:
            importance = importance[0]

        # 选择token
        kept_indices, pruned_indices = self.select_tokens(importance)

        # 统计信息
        num_visual_tokens = visual_token_end - visual_token_start
        num_kept = len(kept_indices)
        stats = {
            "num_visual_tokens": num_visual_tokens,
            "num_kept": num_kept,
            "num_pruned": num_visual_tokens - num_kept,
            "keep_ratio": num_kept / num_visual_tokens,
            "importance_mean": importance.mean().item(),
            "importance_std": importance.std().item(),
            "importance_max": importance.max().item(),
            "importance_min": importance.min().item(),
        }

        if self.verbose:
            print(f"[Stage 2] Token selection:")
            print(f"  Visual tokens: {num_visual_tokens}")
            print(f"  Kept: {num_kept} ({stats['keep_ratio']:.1%})")
            print(f"  Pruned: {stats['num_pruned']}")

        return kept_indices, stats


def register_attention_pruning_hook(
    model: nn.Module,
    selector: AttentionBasedTokenSelector,
    cache: PruneVidDynamicCache,
    visual_token_start: int,
    visual_token_end: int,
    adapter=None,  # 添加adapter参数以获取Stage 1更新后的位置
) -> torch.utils.hooks.RemovableHandle:
    """
    注册attention pruning的forward hook

    在pruning_layer的forward过程中，提取注意力权重，
    计算token重要性，并配置缓存进行压缩。

    Args:
        model: LLM模型 (Qwen2_5_VLForConditionalGeneration)
        selector: Token选择器
        cache: 自定义缓存
        visual_token_start: 视觉token起始位置（原始值）
        visual_token_end: 视觉token结束位置（原始值）
        adapter: Qwen25VLAdapter实例，用于获取Stage 1更新后的位置

    Returns:
        hook_handle: Hook句柄，用于后续移除
    """
    pruning_layer_idx = selector.pruning_layer

    # 获取目标层
    # Qwen2.5-VL结构: model.model.layers (没有 language_model 这一层)
    target_layer = model.model.layers[pruning_layer_idx]

    def pruning_hook(module, input, output):
        """
        Forward hook函数

        output格式（当output_attentions=True时）：
        - output[0]: hidden_states [B, seq_len, hidden_dim]
        - output[1]: attention_weights [B, num_heads, seq_len, seq_len]（可选）
        """
        # 提取注意力权重
        if len(output) < 2:
            # 没有attention_weights，跳过剪枝
            return

        attention_weights = output[1]
        if attention_weights is None:
            return

        # 获取实际的视觉token位置
        # 如果Stage 1执行了压缩，使用更新后的位置
        actual_visual_start = visual_token_start
        actual_visual_end = visual_token_end

        if adapter is not None and hasattr(adapter, 'compressed_visual_token_end'):
            actual_visual_end = adapter.compressed_visual_token_end
            if hasattr(adapter, 'config') and adapter.config.verbose:
                print(f"[Stage 2] Using compressed visual_end: {visual_token_end} -> {actual_visual_end}")

        # 执行token选择
        kept_indices, stats = selector(
            attention_weights,
            actual_visual_start,
            actual_visual_end,
        )

        # 配置缓存进行压缩
        cache.configure_pruning(
            pruning_layer=pruning_layer_idx,
            visual_token_start=actual_visual_start,
            visual_token_end=actual_visual_end,
            kept_visual_indices=kept_indices,
        )

    # 注册hook
    handle = target_layer.register_forward_hook(pruning_hook)

    return handle


def get_pruning_statistics(selector: AttentionBasedTokenSelector) -> Optional[Dict]:
    """
    获取最近一次剪枝的统计信息

    Args:
        selector: Token选择器

    Returns:
        stats: 统计信息，如果没有执行过剪枝则返回None
    """
    if selector.last_kept_indices is None:
        return None

    importance = selector.last_importance_scores
    kept_indices = selector.last_kept_indices

    stats = {
        "num_total": len(importance),
        "num_kept": len(kept_indices),
        "keep_ratio": len(kept_indices) / len(importance),
        "importance_of_kept": {
            "mean": importance[kept_indices].mean().item(),
            "std": importance[kept_indices].std().item(),
            "min": importance[kept_indices].min().item(),
            "max": importance[kept_indices].max().item(),
        },
        "importance_of_pruned": {},
    }

    # 被剪枝token的重要性
    all_indices = torch.arange(len(importance), device=importance.device)
    mask = torch.ones(len(importance), dtype=torch.bool, device=importance.device)
    mask[kept_indices] = False
    pruned_indices = all_indices[mask]

    if len(pruned_indices) > 0:
        stats["importance_of_pruned"] = {
            "mean": importance[pruned_indices].mean().item(),
            "std": importance[pruned_indices].std().item(),
            "min": importance[pruned_indices].min().item(),
            "max": importance[pruned_indices].max().item(),
        }

    return stats
