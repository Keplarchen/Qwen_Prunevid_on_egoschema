"""
Stage 1: 时空Token合并

在视觉编码器输出后，通过以下步骤减少视觉token的冗余：
1. 时序聚类：将视频帧聚类成多个场景段（DPC-KNN + 连续性约束）
2. 静态/动态分离：在每个场景段内，区分静态区域和动态区域
3. 时序合并：对静态token沿时间维度取平均
4. 空间合并：对静态和动态区域分别进行空间聚类和合并

论文3.2节
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List

try:
    from .utils import (
        cluster_dpc_knn,
        detect_static_tokens,
        merge_tokens_by_indices,
        enforce_temporal_continuity,
    )
except ImportError:
    from utils import (
        cluster_dpc_knn,
        detect_static_tokens,
        merge_tokens_by_indices,
        enforce_temporal_continuity,
    )


class SpatialTemporalMerger:
    """
    时空Token合并器

    将视频token进行时空合并，减少冗余
    """

    def __init__(
        self,
        tau: float = 0.8,
        cluster_ratio: float = 0.5,
        temporal_segment_ratio: float = 0.25,
        dpc_knn_k: int = 5,
        verbose: bool = False,
    ):
        """
        Args:
            tau: 静态/动态分离阈值（余弦相似度）
            cluster_ratio: 空间聚类保留比例
            temporal_segment_ratio: 时序分段比例
            dpc_knn_k: DPC-KNN的k近邻参数
            verbose: 是否打印详细信息
        """
        self.tau = tau
        self.cluster_ratio = cluster_ratio
        self.temporal_segment_ratio = temporal_segment_ratio
        self.dpc_knn_k = dpc_knn_k
        self.verbose = verbose

        # 统计信息
        self.stats = {}

    def temporal_clustering(
        self,
        frame_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        对视频帧进行时序聚类

        Args:
            frame_tokens: [T, N, C] T帧，每帧N个token

        Returns:
            cluster_indices: [T] 每帧的聚类索引
            segments: List of frame index lists for each segment
        """
        T, N, C = frame_tokens.shape

        # 计算每帧的平均特征
        frame_features = frame_tokens.mean(dim=1)  # [T, C]

        # 确定聚类数量
        num_clusters = max(1, int(T * self.temporal_segment_ratio))

        # 使用DPC-KNN进行聚类
        cluster_indices, _ = cluster_dpc_knn(
            frame_features,
            cluster_num=num_clusters,
            k=self.dpc_knn_k,
            device=frame_tokens.device,
        )

        # 强制时序连续性
        cluster_indices = enforce_temporal_continuity(cluster_indices)

        # 构建每个segment包含的帧索引
        segments = []
        for cluster_id in range(num_clusters):
            frame_ids = (cluster_indices == cluster_id).nonzero(as_tuple=True)[0].tolist()
            if len(frame_ids) > 0:
                segments.append(frame_ids)

        if self.verbose:
            print(f"[Stage 1] Temporal clustering:")
            print(f"  Frames: {T}")
            print(f"  Segments: {len(segments)}")
            print(f"  Frames per segment: {[len(seg) for seg in segments]}")

        return cluster_indices, segments

    def separate_static_dynamic(
        self,
        segment_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分离静态和动态token

        Args:
            segment_tokens: [T_seg, N, C] 某个segment内的tokens

        Returns:
            static_mask: [N] 布尔mask，True表示静态
            similarity_scores: [N] 每个token的时序相似度
        """
        T_seg, N, C = segment_tokens.shape

        if T_seg <= 1:
            # 只有一帧，全部视为动态
            return torch.zeros(N, dtype=torch.bool, device=segment_tokens.device), \
                   torch.zeros(N, device=segment_tokens.device)

        # 对每个空间位置，计算时序相似度
        similarity_scores = torch.zeros(N, device=segment_tokens.device)

        for i in range(N):
            token_seq = segment_tokens[:, i, :]  # [T_seg, C]
            # 计算所有帧对之间的相似度
            token_seq_norm = torch.nn.functional.normalize(token_seq, p=2, dim=1)
            sim_matrix = torch.mm(token_seq_norm, token_seq_norm.T)  # [T_seg, T_seg]

            # 取上三角的平均值（不包括对角线）
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            if mask.sum() > 0:
                similarity_scores[i] = sim_matrix[mask].mean()

        # 静态token：相似度高于阈值
        static_mask = similarity_scores >= self.tau

        return static_mask, similarity_scores

    def merge_static_tokens_temporally(
        self,
        segment_tokens: torch.Tensor,
        static_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对静态token沿时间维度取平均

        Args:
            segment_tokens: [T_seg, N, C]
            static_mask: [N] 静态token的mask

        Returns:
            merged_static: [N_static, C] 合并后的静态token
        """
        T_seg, N, C = segment_tokens.shape

        if static_mask.sum() == 0:
            # 没有静态token
            return torch.empty(0, C, device=segment_tokens.device)

        # 提取静态token
        static_tokens = segment_tokens[:, static_mask, :]  # [T_seg, N_static, C]

        # 沿时间维度取平均
        merged_static = static_tokens.mean(dim=0)  # [N_static, C]

        return merged_static

    def spatial_clustering_and_merge(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        对token进行空间聚类和合并

        Args:
            tokens: [N, C] 输入token

        Returns:
            merged_tokens: [N_cluster, C] 合并后的token
        """
        N, C = tokens.shape

        if N <= 1:
            return tokens

        # 确定聚类数量
        num_clusters = max(1, int(N * self.cluster_ratio))

        if num_clusters >= N:
            # 不需要聚类
            return tokens

        # 使用DPC-KNN进行空间聚类
        cluster_indices, _ = cluster_dpc_knn(
            tokens,
            cluster_num=num_clusters,
            k=min(self.dpc_knn_k, N - 1),
            device=tokens.device,
        )

        # 合并同一聚类的token
        merged_tokens = merge_tokens_by_indices(
            tokens, cluster_indices, num_clusters, method="mean"
        )

        return merged_tokens

    def process_segment(
        self,
        segment_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        处理单个时序segment

        Args:
            segment_tokens: [T_seg, N, C]

        Returns:
            merged_tokens: [N_merged, C] 合并后的token
            stats: 统计信息
        """
        T_seg, N, C = segment_tokens.shape

        # 1. 分离静态和动态token
        static_mask, similarity_scores = self.separate_static_dynamic(segment_tokens)
        dynamic_mask = ~static_mask

        num_static = static_mask.sum().item()
        num_dynamic = dynamic_mask.sum().item()

        # 2. 对静态token沿时间维度合并
        merged_static = self.merge_static_tokens_temporally(
            segment_tokens, static_mask
        )  # [N_static, C]

        # 3. 对动态token保持所有帧
        dynamic_tokens = segment_tokens[:, dynamic_mask, :]  # [T_seg, N_dynamic, C]
        dynamic_tokens = dynamic_tokens.reshape(-1, C)  # [T_seg * N_dynamic, C]

        # 4. 对静态和动态token分别进行空间聚类
        if merged_static.shape[0] > 0:
            merged_static = self.spatial_clustering_and_merge(merged_static)

        if dynamic_tokens.shape[0] > 0:
            dynamic_tokens = self.spatial_clustering_and_merge(dynamic_tokens)

        # 5. 拼接静态和动态token
        if merged_static.shape[0] > 0 and dynamic_tokens.shape[0] > 0:
            merged_tokens = torch.cat([merged_static, dynamic_tokens], dim=0)
        elif merged_static.shape[0] > 0:
            merged_tokens = merged_static
        else:
            merged_tokens = dynamic_tokens

        # 统计信息
        stats = {
            "frames": T_seg,
            "tokens_per_frame": N,
            "num_static": num_static,
            "num_dynamic": num_dynamic,
            "tokens_before": T_seg * N,
            "tokens_after": merged_tokens.shape[0],
            "reduction_ratio": merged_tokens.shape[0] / (T_seg * N),
        }

        return merged_tokens, stats

    def __call__(
        self,
        frame_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行时空token合并

        Args:
            frame_tokens: [T, N, C] 视频token

        Returns:
            merged_tokens: [N_merged, C] 合并后的token
            stats: 统计信息
        """
        T, N, C = frame_tokens.shape
        device = frame_tokens.device

        # 1. 时序聚类
        cluster_indices, segments = self.temporal_clustering(frame_tokens)

        # 2. 对每个segment进行处理
        all_merged_tokens = []
        segment_stats = []

        for seg_id, frame_ids in enumerate(segments):
            segment_tokens = frame_tokens[frame_ids]  # [T_seg, N, C]
            merged_tokens, seg_stats = self.process_segment(segment_tokens)
            all_merged_tokens.append(merged_tokens)
            segment_stats.append(seg_stats)

        # 3. 拼接所有segment的token
        merged_tokens = torch.cat(all_merged_tokens, dim=0)

        # 总体统计
        total_stats = {
            "frames": T,
            "tokens_per_frame": N,
            "num_segments": len(segments),
            "tokens_before": T * N,
            "tokens_after": merged_tokens.shape[0],
            "reduction_ratio": merged_tokens.shape[0] / (T * N),
            "segment_stats": segment_stats,
        }

        if self.verbose:
            print(f"[Stage 1] Spatial-temporal merging:")
            print(f"  Tokens before: {total_stats['tokens_before']}")
            print(f"  Tokens after: {total_stats['tokens_after']}")
            print(f"  Reduction: {total_stats['reduction_ratio']:.1%}")

        self.stats = total_stats

        return merged_tokens, total_stats


def create_spatial_temporal_merger(
    tau: float = 0.8,
    cluster_ratio: float = 0.5,
    temporal_segment_ratio: float = 0.25,
    dpc_knn_k: int = 5,
    verbose: bool = False,
) -> SpatialTemporalMerger:
    """
    创建时空token合并器

    Args:
        tau: 静态/动态分离阈值
        cluster_ratio: 空间聚类保留比例
        temporal_segment_ratio: 时序分段比例
        dpc_knn_k: DPC-KNN的k近邻参数
        verbose: 是否打印详细信息

    Returns:
        merger: SpatialTemporalMerger实例
    """
    return SpatialTemporalMerger(
        tau=tau,
        cluster_ratio=cluster_ratio,
        temporal_segment_ratio=temporal_segment_ratio,
        dpc_knn_k=dpc_knn_k,
        verbose=verbose,
    )
