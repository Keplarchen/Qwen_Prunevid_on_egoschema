"""
Stage 3: KV缓存压缩
=================

实现PruneVid的第三阶段：在生成阶段压缩KV cache以减少内存和计算开销。

核心思想（参考论文第3.3节）：
在Stage 2选择重要token后，前M层的KV cache中仍包含所有视觉token。
通过压缩这些层的cache，只保留选中的token，可以：
1. 减少内存占用
2. 加速后续的attention计算（序列长度减小）

论文方法：
- 对layers 1 到 M：压缩KV cache
- 对layers M+1 到 L：使用已压缩的序列，自然产生更小的cache

压缩公式（论文公式6-7）：
K̃^(l) = [K̃_v^(l); K_q^(l)]
Ṽ^(l) = [Ṽ_v^(l); V_q^(l)]

其中K̃_v^(l)和Ṽ_v^(l)只包含选中的视觉token。
"""

import torch
from typing import List, Tuple, Optional, Dict
from transformers.cache_utils import DynamicCache


class PruneVidDynamicCache(DynamicCache):
    """
    支持PruneVid的动态KV缓存

    继承自transformers的DynamicCache，添加了压缩功能。

    压缩策略：
    1. 在前M层：存储完整的KV，但在Stage 2完成后立即压缩
    2. 在后L-M层：自动使用压缩后的序列，无需额外处理
    """

    def __init__(self, verbose: bool = False):
        """
        初始化

        Args:
            verbose: 是否输出详细日志
        """
        super().__init__()

        # PruneVid相关配置
        self.pruning_layer = None
        self.visual_token_start = None
        self.visual_token_end = None
        self.selected_visual_indices = None  # 相对于visual_start的索引
        self.kept_all_indices = None  # 全局索引

        # 压缩状态
        self.compressed = False
        self.verbose = verbose

        # 统计信息
        self.compression_stats = {}

    def configure_pruning(
        self,
        pruning_layer: int,
        visual_token_start: int,
        visual_token_end: int,
        selected_visual_indices: torch.Tensor,
    ):
        """
        配置压缩参数（由Stage 2调用）

        Args:
            pruning_layer: 执行剪枝的层索引
            visual_token_start: 视觉token起始位置（Stage 1后的位置）
            visual_token_end: 视觉token结束位置
            selected_visual_indices: 选中的视觉token的全局索引
        """
        self.pruning_layer = pruning_layer
        self.visual_token_start = visual_token_start
        self.visual_token_end = visual_token_end

        # 计算相对索引
        self.selected_visual_indices = selected_visual_indices - visual_token_start

        # 构建完整的保留索引列表
        # [文本前] + [选中的视觉] + [文本后]
        # 注意：这里需要知道原始序列长度，我们先存储配置，实际压缩时再构建

        if self.verbose:
            print(f"[Stage 3] 配置KV缓存压缩:")
            print(f"  剪枝层: {pruning_layer}")
            print(f"  视觉token范围: [{visual_token_start}, {visual_token_end})")
            print(f"  选中: {len(selected_visual_indices)} 个视觉tokens")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache（重写父类方法）

        在经过pruning_layer时，触发压缩。

        Args:
            key_states: [batch, num_heads, seq_len, head_dim] 新的key
            value_states: [batch, num_heads, seq_len, head_dim] 新的value
            layer_idx: 当前层索引
            cache_kwargs: 额外参数

        Returns:
            updated_key: 更新后的完整key
            updated_value: 更新后的完整value
        """
        # 调用父类的update方法
        updated_key, updated_value = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        # 检查是否需要压缩
        # 压缩时机：刚完成pruning_layer，且尚未压缩
        if (
            self.pruning_layer is not None
            and layer_idx == self.pruning_layer
            and not self.compressed
            and self.selected_visual_indices is not None
        ):
            # 执行压缩
            self._compress_cache()
            self.compressed = True

            # 返回压缩后的当前层KV
            updated_key = self.key_cache[layer_idx]
            updated_value = self.value_cache[layer_idx]

        return updated_key, updated_value

    def _compress_cache(self):
        """
        压缩前M+1层的KV cache

        只保留：
        - 所有文本token
        - 选中的视觉token
        """
        if self.visual_token_start is None:
            if self.verbose:
                print("[Stage 3] 警告：视觉token位置未设置，跳过压缩")
            return

        # 获取当前缓存的序列长度（从第0层的key获取）
        if len(self.key_cache) == 0:
            if self.verbose:
                print("[Stage 3] 警告：缓存为空，跳过压缩")
            return

        # 示例key shape: [batch, num_heads, seq_len, head_dim]
        original_seq_len = self.key_cache[0].shape[2]

        # 构建保留的索引
        # [0, ..., visual_start-1] + [selected visual] + [visual_end, ..., seq_len-1]
        device = self.key_cache[0].device

        text_before = torch.arange(0, self.visual_token_start, device=device)
        selected_visual = self.selected_visual_indices + self.visual_token_start
        text_after = torch.arange(self.visual_token_end, original_seq_len, device=device)

        # 合并
        kept_indices = torch.cat([text_before, selected_visual, text_after], dim=0)

        new_seq_len = kept_indices.shape[0]

        if self.verbose:
            print(f"[Stage 3] 压缩KV cache:")
            print(f"  原始序列长度: {original_seq_len}")
            print(f"  压缩后序列长度: {new_seq_len}")
            print(f"  压缩层数: 0 到 {self.pruning_layer}")

        # 压缩layers 0 到 pruning_layer
        for layer_idx in range(self.pruning_layer + 1):
            if layer_idx < len(self.key_cache):
                # key_cache[layer_idx]: [batch, num_heads, seq_len, head_dim]
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, kept_indices, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, kept_indices, :]

                if self.verbose and layer_idx == 0:
                    print(f"  示例：Layer 0 KV shape: {self.key_cache[0].shape}")

        # 记录统计
        self.compression_stats = {
            "original_seq_len": original_seq_len,
            "compressed_seq_len": new_seq_len,
            "num_compressed_layers": self.pruning_layer + 1,
            "reduction_ratio": (original_seq_len - new_seq_len) / original_seq_len,
        }

        if self.verbose:
            print(f"[Stage 3] KV缓存压缩完成")

    def get_compression_stats(self) -> Dict:
        """
        获取压缩统计信息

        Returns:
            stats: 压缩统计字典
        """
        return self.compression_stats

    def reset(self):
        """重置cache和压缩状态"""
        super().reset()
        self.compressed = False
        self.pruning_layer = None
        self.visual_token_start = None
        self.visual_token_end = None
        self.selected_visual_indices = None
        self.kept_all_indices = None
        self.compression_stats = {}

        if self.verbose:
            print("[Stage 3] KV缓存已重置")


# 测试示例（可选）
def test_prunevid_cache():
    """测试PruneVidDynamicCache"""
    print("Testing PruneVidDynamicCache...")

    cache = PruneVidDynamicCache(verbose=True)

    # 配置压缩
    cache.configure_pruning(
        pruning_layer=2,
        visual_token_start=5,
        visual_token_end=15,
        selected_visual_indices=torch.tensor([5, 7, 9, 11]),  # 全局索引
    )

    # 模拟添加KV cache
    batch_size = 1
    num_heads = 8
    seq_len = 20
    head_dim = 64

    for layer_idx in range(5):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        updated_key, updated_value = cache.update(key, value, layer_idx)

        print(f"\nLayer {layer_idx}:")
        print(f"  Key shape: {updated_key.shape}")
        print(f"  Value shape: {updated_value.shape}")

    # 检查压缩统计
    stats = cache.get_compression_stats()
    print(f"\n压缩统计: {stats}")

    print("\nPruneVidDynamicCache test passed!")


if __name__ == "__main__":
    test_prunevid_cache()


# 导出
__all__ = ["PruneVidDynamicCache"]
