"""
PruneVid Configuration

包含PruneVid方法的所有超参数配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PruneVidConfig:
    """
    PruneVid方法的配置类

    三个阶段的超参数：
    - Stage 1: 时空token合并（视觉编码器级别）
    - Stage 2: 基于注意力的token选择（LLM中间层）
    - Stage 3: KV缓存压缩（解码阶段）
    """

    # ========== 通用配置 ==========
    device: str = "cuda"
    verbose: bool = False

    # ========== Stage 1: 时空Token合并 ==========
    enable_stage1: bool = True

    # 静态/动态token分离阈值（论文中tau=0.8）
    # 当token之间的余弦相似度 > tau时，认为是静态区域
    tau: float = 0.8

    # 空间聚类保留比例（论文中为0.5）
    # 对静态和动态区域分别进行空间聚类，保留50%的token
    cluster_ratio: float = 0.5

    # 时序分段比例（论文中为0.25）
    # 将视频分成 T * temporal_segment_ratio 个时序段
    temporal_segment_ratio: float = 0.25

    # DPC-KNN聚类参数
    dpc_knn_k: int = 5  # k近邻数量

    # ========== Stage 2: 基于注意力的Token选择 ==========
    enable_pruning: bool = True

    # Token保留比例（论文中alpha=0.4，即保留40%）
    keep_ratio: float = 0.4

    # 在哪一层进行剪枝（论文中M=10）
    # 对于Qwen2.5-VL-7B（28层），默认在第10层
    pruning_layer: int = 10

    # 注意力聚合策略：'max' 或 'mean'
    # 论文使用max-max策略：先对问题token取max，再对注意力头取max
    attention_aggregation: str = "max"

    # ========== Stage 3: KV缓存压缩 ==========
    # Stage 3自动启用（当enable_pruning=True时）
    # KV缓存会对前pruning_layer层进行压缩

    # ========== 视频处理配置 ==========
    # 最大帧数（论文中PLLaVA使用16帧，LLaVA-OneVision使用32帧）
    max_frames: int = 16

    # 视频采样方式: 'uniform' 或 'fps'
    video_sampling: str = "uniform"

    # 最小短边分辨率
    min_pixels: int = 224 * 224
    max_pixels: int = 1280 * 28 * 28

    # ========== 生成配置 ==========
    max_new_tokens: int = 100
    temperature: float = 0.0  # 0表示greedy decoding
    do_sample: bool = False

    def validate(self):
        """验证配置的有效性"""
        assert 0.0 < self.tau <= 1.0, "tau应在(0, 1]范围内"
        assert 0.0 < self.cluster_ratio <= 1.0, "cluster_ratio应在(0, 1]范围内"
        assert 0.0 < self.temporal_segment_ratio <= 1.0, "temporal_segment_ratio应在(0, 1]范围内"
        assert 0.0 < self.keep_ratio <= 1.0, "keep_ratio应在(0, 1]范围内"
        assert self.pruning_layer > 0, "pruning_layer应为正整数"
        assert self.attention_aggregation in ["max", "mean"], "attention_aggregation应为'max'或'mean'"
        assert self.max_frames > 0, "max_frames应为正整数"

    def get_stage_status(self) -> dict:
        """返回各阶段的启用状态"""
        return {
            "stage1_merging": self.enable_stage1,
            "stage2_selection": self.enable_pruning,
            "stage3_cache": self.enable_pruning,  # Stage 3跟随Stage 2
        }

    def __str__(self) -> str:
        """打印配置信息"""
        lines = ["PruneVid Configuration:"]
        lines.append(f"  Stage 1 (Spatial-Temporal Merging): {'ON' if self.enable_stage1 else 'OFF'}")
        if self.enable_stage1:
            lines.append(f"    - tau: {self.tau}")
            lines.append(f"    - cluster_ratio: {self.cluster_ratio}")
            lines.append(f"    - temporal_segment_ratio: {self.temporal_segment_ratio}")

        lines.append(f"  Stage 2 (Attention-based Selection): {'ON' if self.enable_pruning else 'OFF'}")
        if self.enable_pruning:
            lines.append(f"    - keep_ratio: {self.keep_ratio}")
            lines.append(f"    - pruning_layer: {self.pruning_layer}")
            lines.append(f"    - attention_aggregation: {self.attention_aggregation}")

        lines.append(f"  Stage 3 (KV Cache Compression): {'ON' if self.enable_pruning else 'OFF'}")

        return "\n".join(lines)


# 预定义配置
def get_baseline_config() -> PruneVidConfig:
    """基线配置：不使用任何剪枝"""
    return PruneVidConfig(
        enable_stage1=False,
        enable_pruning=False,
    )


def get_paper_config() -> PruneVidConfig:
    """论文推荐配置（表2，表5）"""
    return PruneVidConfig(
        enable_stage1=True,
        tau=0.8,
        cluster_ratio=0.5,
        temporal_segment_ratio=0.25,
        enable_pruning=True,
        keep_ratio=0.4,
        pruning_layer=10,
    )


def get_conservative_config() -> PruneVidConfig:
    """保守配置：较少的剪枝，更高的准确率"""
    return PruneVidConfig(
        enable_stage1=True,
        tau=0.9,  # 更高的阈值，更少的静态token
        cluster_ratio=0.7,  # 保留更多token
        temporal_segment_ratio=0.25,
        enable_pruning=True,
        keep_ratio=0.6,  # 保留60%
        pruning_layer=10,
    )


def get_aggressive_config() -> PruneVidConfig:
    """激进配置：更多的剪枝，更高的效率"""
    return PruneVidConfig(
        enable_stage1=True,
        tau=0.7,  # 更低的阈值，更多的静态token
        cluster_ratio=0.3,  # 保留更少token
        temporal_segment_ratio=0.25,
        enable_pruning=True,
        keep_ratio=0.3,  # 仅保留30%
        pruning_layer=10,
    )
