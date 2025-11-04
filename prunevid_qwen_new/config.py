"""
PruneVid配置类
==============

定义PruneVid方法的所有超参数和配置选项。

参考论文参数设置：
- tau = 0.8 (静态/动态token分离阈值)
- cluster_ratio = 0.5 (空间聚类保留比例)
- temporal_segment_ratio = 0.25 (时序分段比例)
- keep_ratio = 0.4 (Stage 2 token保留比例，论文中的α)
- pruning_layer = 10 (对于28层的Qwen2.5-VL-7B)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PruneVidConfig:
    """PruneVid方法的完整配置"""

    # ==================== Stage 1: 时空Token合并 ====================

    # 静态token检测阈值（余弦相似度）
    # 论文：τ = 0.8，越高越严格（需要更相似才认为是静态）
    tau: float = 0.8

    # 空间聚类后的保留比例
    # 论文：0.5，即每个聚类合并后保留50%的token数量
    cluster_ratio: float = 0.5

    # 时序分段比例
    # 论文：0.25，用于DPC-KNN聚类时的密度计算
    # 较小的值会产生更多的时序段（更细粒度）
    temporal_segment_ratio: float = 0.25

    # DPC-KNN算法的K值（K近邻数量）
    # 论文：k = 5
    dpc_knn_k: int = 5

    # 是否启用Stage 1
    enable_stage1: bool = True

    # ==================== Stage 2: 注意力选择 ====================

    # Token保留比例（论文中的α）
    # 论文：α = 0.4，即保留40%最重要的视觉token
    keep_ratio: float = 0.4

    # 执行token剪枝的层索引（论文中的M）
    # 对于28层的Qwen2.5-VL-7B，论文使用第10层
    # 对于32层的模型，可以使用第10-12层
    pruning_layer: int = 10

    # 注意力聚合策略："max"（max-max）或"mean"
    # 论文使用max-max：先对问题token维度取max，再对注意力头维度取max
    attention_aggregation: str = "max"

    # 是否启用Stage 2
    enable_stage2: bool = True

    # ==================== Stage 3: KV缓存压缩 ====================

    # 是否启用KV缓存压缩
    # 论文：启用，对前M层的缓存进行压缩
    enable_cache_compression: bool = True

    # ==================== 视频处理 ====================

    # 最大帧数
    max_frames: int = 100

    # 视频采样方式："uniform"（均匀采样）或"fps"（基于FPS）
    video_sampling: str = "uniform"

    # 最小像素数（用于视频分辨率规范化）
    min_pixels: int = 224 * 224

    # 最大像素数
    max_pixels: int = 1280 * 28 * 28

    # ==================== 调试和日志 ====================

    # 是否输出详细日志
    verbose: bool = False

    # 是否收集统计信息（token数量、压缩比等）
    collect_stats: bool = True

    def __post_init__(self):
        """验证配置参数的有效性"""
        assert 0 <= self.tau <= 1, f"tau必须在[0,1]范围内，当前值：{self.tau}"
        assert 0 < self.cluster_ratio <= 1, f"cluster_ratio必须在(0,1]范围内，当前值：{self.cluster_ratio}"
        assert 0 < self.temporal_segment_ratio <= 1, f"temporal_segment_ratio必须在(0,1]范围内，当前值：{self.temporal_segment_ratio}"
        assert self.dpc_knn_k >= 1, f"dpc_knn_k必须>=1，当前值：{self.dpc_knn_k}"
        assert 0 < self.keep_ratio <= 1, f"keep_ratio必须在(0,1]范围内，当前值：{self.keep_ratio}"
        assert self.pruning_layer >= 0, f"pruning_layer必须>=0，当前值：{self.pruning_layer}"
        assert self.attention_aggregation in ["max", "mean"], \
            f"attention_aggregation必须是'max'或'mean'，当前值：{self.attention_aggregation}"

    def to_dict(self):
        """转换为字典格式"""
        return {
            # Stage 1
            "tau": self.tau,
            "cluster_ratio": self.cluster_ratio,
            "temporal_segment_ratio": self.temporal_segment_ratio,
            "dpc_knn_k": self.dpc_knn_k,
            "enable_stage1": self.enable_stage1,
            # Stage 2
            "keep_ratio": self.keep_ratio,
            "pruning_layer": self.pruning_layer,
            "attention_aggregation": self.attention_aggregation,
            "enable_stage2": self.enable_stage2,
            # Stage 3
            "enable_cache_compression": self.enable_cache_compression,
            # Video
            "max_frames": self.max_frames,
            "video_sampling": self.video_sampling,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            # Debug
            "verbose": self.verbose,
            "collect_stats": self.collect_stats,
        }


def get_baseline_config() -> PruneVidConfig:
    """
    基线配置：禁用所有剪枝，用于性能对比

    Returns:
        PruneVidConfig: 禁用所有stage的配置
    """
    return PruneVidConfig(
        enable_stage1=False,
        enable_stage2=False,
        enable_cache_compression=False,
        verbose=False,
        collect_stats=True,
    )


def get_paper_config() -> PruneVidConfig:
    """
    论文推荐配置：按照ACL 2025论文的设置

    在PLLaVA上的性能：
    - Token保留率：16.2%
    - FLOPs：0.23×
    - MVBench准确率：47.6（基线46.6）
    - VideoMME：45.0（基线44.4）
    - EgoSchema：49.0/42.6（基线47.8/42.6）

    Returns:
        PruneVidConfig: 论文推荐的配置
    """
    return PruneVidConfig(
        # Stage 1
        tau=0.8,
        cluster_ratio=0.5,
        temporal_segment_ratio=0.25,
        dpc_knn_k=5,
        enable_stage1=True,
        # Stage 2
        keep_ratio=0.4,
        pruning_layer=10,
        attention_aggregation="max",
        enable_stage2=True,
        # Stage 3
        enable_cache_compression=True,
        # Video
        max_frames=16,
        video_sampling="uniform",
        # Debug
        verbose=False,
        collect_stats=True,
    )


def get_conservative_config() -> PruneVidConfig:
    """
    保守配置：更激进的压缩，追求更高的效率

    适用场景：
    - 资源受限环境
    - 对速度要求高
    - 可以容忍轻微的准确率下降

    Returns:
        PruneVidConfig: 保守（高压缩）配置
    """
    return PruneVidConfig(
        # Stage 1: 更激进的合并
        tau=0.7,  # 更低的阈值，更多token被认为是静态的
        cluster_ratio=0.3,  # 更小的保留比例
        temporal_segment_ratio=0.25,
        dpc_knn_k=5,
        enable_stage1=True,
        # Stage 2: 保留更少的token
        keep_ratio=0.3,  # 只保留30%
        pruning_layer=10,
        attention_aggregation="max",
        enable_stage2=True,
        # Stage 3
        enable_cache_compression=True,
        # Video
        max_frames=16,
        video_sampling="uniform",
        # Debug
        verbose=False,
        collect_stats=True,
    )


def get_aggressive_config() -> PruneVidConfig:
    """
    激进配置：较轻的压缩，追求更高的准确率

    适用场景：
    - 资源充足
    - 对准确率要求高
    - 仍希望获得一定的效率提升

    Returns:
        PruneVidConfig: 激进（低压缩）配置
    """
    return PruneVidConfig(
        # Stage 1: 较轻的合并
        tau=0.85,  # 更高的阈值，更严格的静态检测
        cluster_ratio=0.6,  # 更大的保留比例
        temporal_segment_ratio=0.25,
        dpc_knn_k=5,
        enable_stage1=True,
        # Stage 2: 保留更多的token
        keep_ratio=0.5,  # 保留50%
        pruning_layer=10,
        attention_aggregation="max",
        enable_stage2=True,
        # Stage 3
        enable_cache_compression=True,
        # Video
        max_frames=16,
        video_sampling="uniform",
        # Debug
        verbose=False,
        collect_stats=True,
    )


def get_stage1_only_config() -> PruneVidConfig:
    """
    仅Stage 1配置：只启用时空合并，用于消融实验

    Returns:
        PruneVidConfig: 仅Stage 1的配置
    """
    return PruneVidConfig(
        # Stage 1
        tau=0.8,
        cluster_ratio=0.5,
        temporal_segment_ratio=0.25,
        dpc_knn_k=5,
        enable_stage1=True,
        # Stage 2: 禁用
        enable_stage2=False,
        # Stage 3: 禁用
        enable_cache_compression=False,
        # Video
        max_frames=16,
        video_sampling="uniform",
        # Debug
        verbose=True,  # 调试时显示详细信息
        collect_stats=True,
    )


def get_high_frame_config(max_frames: int = 100) -> PruneVidConfig:
    """
    高帧率配置：优化用于处理100+帧的视频
    使用更激进的压缩策略以控制内存使用

    Args:
        max_frames: 最大帧数，默认100

    Returns:
        PruneVidConfig: 针对高帧率优化的配置
    """
    return PruneVidConfig(
        # Stage 1: 更激进的时空合并
        tau=0.75,  # 更低阈值 = 检测更多静态token
        cluster_ratio=0.4,  # 每个cluster保留更少token
        temporal_segment_ratio=0.2,  # 更细粒度的时间分段
        dpc_knn_k=5,
        enable_stage1=True,
        # Stage 2: 激进的token剪枝
        keep_ratio=0.3,  # 只保留30%的token
        pruning_layer=10,
        attention_aggregation="max",
        enable_stage2=True,
        # Stage 3: 启用缓存压缩
        enable_cache_compression=True,
        # Video: 高帧率
        max_frames=max_frames,
        video_sampling="uniform",
        # Debug
        verbose=True,
        collect_stats=True,
    )


# 导出所有配置获取函数
__all__ = [
    "PruneVidConfig",
    "get_baseline_config",
    "get_paper_config",
    "get_conservative_config",
    "get_aggressive_config",
    "get_stage1_only_config",
    "get_high_frame_config",
]
