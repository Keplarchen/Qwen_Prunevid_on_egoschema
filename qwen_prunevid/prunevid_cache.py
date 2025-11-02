"""
PruneVid自定义Cache实现
支持在指定layer自动压缩KV cache，真正删除被剪枝的token
"""

import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, Tuple


class PruneVidDynamicCache(DynamicCache):
    """
    支持PruneVid token pruning的DynamicCache实现

    核心机制：
    1. 在pruning_layer完成forward时，根据kept_visual_indices压缩KV cache
    2. 真正删除token（减少seq_len维度），而非mask置零
    3. 压缩所有已完成的层（0 到 pruning_layer）

    与标准DynamicCache的区别：
    - 标准：每层cache独立，只能append
    - PruneVid：在指定layer触发全局压缩，实际删除token
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pruning配置（由adapter设置）
        self.pruning_enabled = False
        self.pruning_layer = None  # 在哪一层应用pruning
        self.kept_visual_indices = None  # 要保留的视觉token索引（相对位置）
        self.visual_range = None  # (visual_start, visual_end) 视觉token的绝对位置
        self.pruning_applied = False  # 是否已经应用过pruning

        # 统计信息
        self.tokens_before_pruning = 0
        self.tokens_after_pruning = 0
        self.num_visual_tokens = 0
        self.num_visual_kept = 0

    def configure_pruning(
        self,
        pruning_layer: int,
        kept_visual_indices: torch.Tensor,
        visual_start: int,
        visual_end: int
    ):
        """
        配置pruning参数（由adapter的hook调用）

        Args:
            pruning_layer: 在第几层应用pruning（0-indexed）
            kept_visual_indices: 要保留的视觉token索引（相对于visual_start的位置）
            visual_start: 视觉token起始位置
            visual_end: 视觉token结束位置（不包含）
        """
        self.pruning_enabled = True
        self.pruning_layer = pruning_layer
        self.kept_visual_indices = kept_visual_indices
        self.visual_range = (visual_start, visual_end)
        self.pruning_applied = False

        self.num_visual_tokens = visual_end - visual_start
        self.num_visual_kept = len(kept_visual_indices)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新cache，在pruning_layer时自动应用token压缩

        Returns:
            keys, values: 更新后的完整keys和values
        """
        # 1. 先调用父类的正常update逻辑
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 2. 检查是否需要应用pruning
        should_apply_pruning = (
            self.pruning_enabled and
            layer_idx == self.pruning_layer and
            not self.pruning_applied and
            self.kept_visual_indices is not None
        )

        if not should_apply_pruning:
            return keys, values

        # 3. 应用pruning：真正删除被剪枝的token
        visual_start, visual_end = self.visual_range
        device = keys.device
        seq_len = keys.shape[2]  # [B, num_heads, seq_len, head_dim]

        self.tokens_before_pruning = seq_len

        # 4. 构建要保留的索引
        # kept_visual_indices 是相对于 visual_start 的位置
        # 需要转换为绝对位置
        kept_visual_abs = self.kept_visual_indices + visual_start

        # 保留所有文本token（visual_end之后的所有token）
        text_indices = torch.arange(visual_end, seq_len, device=device)

        # 最终保留：[kept_visual_tokens, all_text_tokens]
        all_kept_indices = torch.cat([kept_visual_abs, text_indices])

        self.tokens_after_pruning = len(all_kept_indices)

        # 5. 压缩所有已完成层的KV cache（0 到 pruning_layer，包含）
        for lid in range(layer_idx + 1):
            # 获取当前层的keys和values
            old_keys = self.layers[lid].keys
            old_values = self.layers[lid].values

            # 关键：使用索引选择，真正减少seq_len维度
            # old_keys: [B, num_heads, seq_len, head_dim]
            # new_keys: [B, num_heads, len(all_kept_indices), head_dim]
            new_keys = old_keys[:, :, all_kept_indices, :].contiguous()
            new_values = old_values[:, :, all_kept_indices, :].contiguous()

            # 更新layer的cache
            self.layers[lid].keys = new_keys
            self.layers[lid].values = new_values

        self.pruning_applied = True

        # 6. 返回当前层压缩后的keys/values
        # 注意：已经被更新了，所以直接返回layers[layer_idx]的
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def get_pruning_stats(self) -> dict:
        """
        获取pruning统计信息

        Returns:
            dict: 包含tokens_before, tokens_after, pruning_ratio等信息
        """
        if not self.pruning_applied:
            return {
                'pruning_applied': False,
                'tokens_before': 0,
                'tokens_after': 0,
                'pruning_ratio': 0.0,
                'num_visual_tokens': 0,
                'num_visual_kept': 0,
                'visual_pruning_ratio': 0.0
            }

        visual_pruning_ratio = 1.0 - (self.num_visual_kept / self.num_visual_tokens) if self.num_visual_tokens > 0 else 0.0
        overall_pruning_ratio = 1.0 - (self.tokens_after_pruning / self.tokens_before_pruning) if self.tokens_before_pruning > 0 else 0.0

        return {
            'pruning_applied': True,
            'tokens_before': self.tokens_before_pruning,
            'tokens_after': self.tokens_after_pruning,
            'pruning_ratio': overall_pruning_ratio,
            'num_visual_tokens': self.num_visual_tokens,
            'num_visual_kept': self.num_visual_kept,
            'visual_pruning_ratio': visual_pruning_ratio
        }

    def reset_pruning_state(self):
        """重置pruning状态（用于新的generation）"""
        self.pruning_enabled = False
        self.pruning_layer = None
        self.kept_visual_indices = None
        self.visual_range = None
        self.pruning_applied = False

        self.tokens_before_pruning = 0
        self.tokens_after_pruning = 0
        self.num_visual_tokens = 0
        self.num_visual_kept = 0
