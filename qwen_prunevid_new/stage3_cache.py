"""
Stage 3: KV缓存压缩

实现自定义的DynamicCache，在剪枝后真正删除被剪枝的token，
而不是仅仅masked掉。这样可以节省内存和计算。

论文3.3节：压缩前M层的KV缓存，保留选中的视觉token和所有文本token
"""

import torch
from typing import List, Optional, Tuple
from transformers import DynamicCache


class PruneVidDynamicCache(DynamicCache):
    """
    PruneVid自定义KV缓存

    在pruning_layer进行剪枝后，对前pruning_layer+1层的KV缓存进行压缩：
    - 保留所有文本token
    - 仅保留选中的视觉token（根据注意力重要性）
    - 真正删除被剪枝的token，而不是mask

    这样可以：
    1. 减少内存占用
    2. 加速后续层的计算（序列长度变短）
    """

    def __init__(self):
        super().__init__()
        # 剪枝配置
        self.pruning_layer: Optional[int] = None
        self.visual_token_start: Optional[int] = None
        self.visual_token_end: Optional[int] = None
        self.kept_visual_indices: Optional[torch.Tensor] = None
        self.is_configured: bool = False
        self.compressed: bool = False

    def configure_pruning(
        self,
        pruning_layer: int,
        visual_token_start: int,
        visual_token_end: int,
        kept_visual_indices: torch.Tensor,
    ):
        """
        配置剪枝参数

        Args:
            pruning_layer: 在哪一层进行剪枝
            visual_token_start: 视觉token的起始位置
            visual_token_end: 视觉token的结束位置
            kept_visual_indices: 保留的视觉token索引（相对于视觉token起始位置）
        """
        self.pruning_layer = pruning_layer
        self.visual_token_start = visual_token_start
        self.visual_token_end = visual_token_end
        self.kept_visual_indices = kept_visual_indices
        self.is_configured = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV缓存

        如果当前层是pruning_layer，则在更新后触发压缩。
        压缩会删除前pruning_layer+1层中被剪枝的视觉token。

        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: 当前层索引
            cache_kwargs: 额外参数

        Returns:
            key_states: 更新后的key
            value_states: 更新后的value
        """
        # 调用父类的update方法，将KV存入缓存
        key_states, value_states = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        # 如果是pruning_layer且配置了剪枝，则进行压缩
        if (
            self.is_configured
            and not self.compressed
            and layer_idx == self.pruning_layer
        ):
            self._compress_cache()
            self.compressed = True

        return key_states, value_states

    def _compress_cache(self):
        """
        压缩KV缓存

        对前pruning_layer+1层（0到pruning_layer）的缓存：
        - 保留所有文本token
        - 仅保留选中的视觉token
        - 删除被剪枝的视觉token
        """
        if not self.is_configured:
            return

        num_layers_to_compress = self.pruning_layer + 1

        # 构建最终保留的token索引
        # [0, visual_token_start) + kept_visual_tokens + [visual_token_end, seq_len)
        kept_indices = self._build_kept_indices()

        # 对每一层进行压缩
        for layer_idx in range(min(num_layers_to_compress, len(self.layers))):
            # 压缩key cache
            if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], 'keys'):
                self.layers[layer_idx].keys = self.layers[layer_idx].keys[
                    :, :, kept_indices, :
                ].contiguous()

            # 压缩value cache
            if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], 'values'):
                self.layers[layer_idx].values = self.layers[layer_idx].values[
                    :, :, kept_indices, :
                ].contiguous()

    def _build_kept_indices(self) -> torch.Tensor:
        """
        构建保留的token索引列表

        Returns:
            kept_indices: 保留的token索引 [num_kept_tokens]
        """
        device = self.kept_visual_indices.device

        # 文本token之前的部分（如果有）
        before_visual = torch.arange(
            self.visual_token_start, device=device, dtype=torch.long
        )

        # 保留的视觉token（需要加上offset）
        kept_visual = self.kept_visual_indices + self.visual_token_start

        # 视觉token之后的部分（如果有）
        # 注意：这里需要获取当前的序列长度
        current_seq_len = self.layers[0].keys.shape[2] if len(self.layers) > 0 and hasattr(self.layers[0], 'keys') else 0
        after_visual = torch.arange(
            self.visual_token_end, current_seq_len, device=device, dtype=torch.long
        )

        # 拼接所有保留的索引
        kept_indices = torch.cat([before_visual, kept_visual, after_visual])

        # 排序以保持顺序
        kept_indices = kept_indices.sort()[0]

        return kept_indices

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """
        获取序列长度

        Args:
            layer_idx: 层索引，如果为None则返回最后一层的长度

        Returns:
            seq_length: 序列长度
        """
        if len(self.layers) == 0:
            return 0

        if layer_idx is None:
            layer_idx = len(self.layers) - 1

        # 使用父类的 get_seq_length 方法
        if layer_idx >= len(self.layers):
            return 0

        return self.layers[layer_idx].get_seq_length()

    def get_compression_ratio(self) -> float:
        """
        获取压缩比例

        Returns:
            compression_ratio: 压缩后序列长度 / 原始序列长度
        """
        if not self.compressed or not self.is_configured:
            return 1.0

        # 原始视觉token数量
        original_visual_tokens = self.visual_token_end - self.visual_token_start
        # 保留的视觉token数量
        kept_visual_tokens = len(self.kept_visual_indices)

        # 计算压缩比
        compression_ratio = kept_visual_tokens / original_visual_tokens

        return compression_ratio

    def get_statistics(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            stats: 统计信息字典
        """
        stats = {
            "num_layers": len(self.layers),
            "compressed": self.compressed,
            "pruning_configured": self.is_configured,
        }

        if self.is_configured:
            stats.update({
                "pruning_layer": self.pruning_layer,
                "visual_token_start": self.visual_token_start,
                "visual_token_end": self.visual_token_end,
                "original_visual_tokens": self.visual_token_end - self.visual_token_start,
                "kept_visual_tokens": len(self.kept_visual_indices),
                "compression_ratio": self.get_compression_ratio(),
            })

        if len(self.layers) > 0:
            # 不同层的序列长度（压缩后前M层会变短）
            seq_lengths = [self.get_seq_length(i) for i in range(len(self.layers))]
            stats["seq_lengths_per_layer"] = seq_lengths

        return stats

    def reset(self):
        """重置缓存"""
        # 清空所有层的缓存
        for layer in self.layers:
            if hasattr(layer, 'keys'):
                layer.keys = torch.tensor([], dtype=layer.dtype, device=layer.device) if hasattr(layer, 'dtype') else torch.tensor([])
            if hasattr(layer, 'values'):
                layer.values = torch.tensor([], dtype=layer.dtype, device=layer.device) if hasattr(layer, 'dtype') else torch.tensor([])
            layer.is_initialized = False

        # 重置剪枝配置
        self.pruning_layer = None
        self.visual_token_start = None
        self.visual_token_end = None
        self.kept_visual_indices = None
        self.is_configured = False
        self.compressed = False


def create_prunevid_cache() -> PruneVidDynamicCache:
    """
    创建PruneVid自定义缓存

    Returns:
        cache: PruneVidDynamicCache实例
    """
    return PruneVidDynamicCache()
