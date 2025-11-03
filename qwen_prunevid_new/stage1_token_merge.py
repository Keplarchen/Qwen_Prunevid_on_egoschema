"""
Stage 1: 时空Token合并（在位置编码之后执行）

在生成 position_ids 和 position_embeddings 之后，对视觉 tokens 进行时空合并：
1. 时序聚类：将视频帧聚类成多个场景段
2. 静态/动态分离：在每个场景段内，区分静态区域和动态区域
3. 静态 token 合并：
   - 时间位置编码：取该时间段内的平均值
   - 空间位置编码：取该位置在时间段内的平均值
4. 动态 token 聚类：
   - 在每一帧中对动态 token 做聚类
   - 时间位置编码：该帧的时间位置编码
   - 空间位置编码：该簇中包含的 token 的平均值

论文3.2节
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
import torch.nn.functional as F

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


class SpatialTemporalTokenMerger:
    """
    时空Token合并器（在位置编码之后执行）

    输入：
        - hidden_states: [batch_size, seq_len, hidden_dim]
        - position_ids: [3, batch_size, seq_len]  # (time, height, width)
        - position_embeddings: Tuple of (pos_emb1, pos_emb2)
            - pos_emb1: [3, batch_size, seq_len, emb_dim]
            - pos_emb2: [3, batch_size, seq_len, emb_dim]
        - video_grid_thw: [[t, h, w], ...]
        - visual_token_indices: List of visual token positions

    输出：
        - merged_hidden_states: [batch_size, new_seq_len, hidden_dim]
        - merged_position_ids: [3, batch_size, new_seq_len]
        - merged_position_embeddings: Tuple of merged pos_emb1, pos_emb2
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
        frame_time_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        对视频帧进行时序聚类

        Args:
            frame_tokens: [T, N, C] T帧，每帧N个token
            frame_time_ids: [T] 每帧的时间ID

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
            token_seq_norm = F.normalize(token_seq, p=2, dim=1)
            sim_matrix = torch.mm(token_seq_norm, token_seq_norm.T)  # [T_seg, T_seg]

            # 取上三角的平均值（不包括对角线）
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            if mask.sum() > 0:
                similarity_scores[i] = sim_matrix[mask].mean()

        # 静态token：相似度高于阈值
        static_mask = similarity_scores >= self.tau

        return static_mask, similarity_scores

    def merge_static_tokens(
        self,
        segment_tokens: torch.Tensor,
        segment_time_ids: torch.Tensor,
        segment_height_ids: torch.Tensor,
        segment_width_ids: torch.Tensor,
        segment_pos_emb1: torch.Tensor,
        segment_pos_emb2: torch.Tensor,
        static_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对静态token进行合并（时间维度平均）

        Args:
            segment_tokens: [T_seg, N, C]
            segment_time_ids: [T_seg, N] 时间位置ID
            segment_height_ids: [T_seg, N] 高度位置ID
            segment_width_ids: [T_seg, N] 宽度位置ID
            segment_pos_emb1: [T_seg, N, emb_dim] 位置编码1
            segment_pos_emb2: [T_seg, N, emb_dim] 位置编码2
            static_mask: [N] 静态token的mask

        Returns:
            merged_static: [N_static, C]
            merged_time_ids: [N_static]
            merged_height_ids: [N_static]
            merged_width_ids: [N_static]
            merged_pos_emb1: [N_static, emb_dim]
            merged_pos_emb2: [N_static, emb_dim]
        """
        T_seg, N, C = segment_tokens.shape

        if static_mask.sum() == 0:
            # 没有静态token
            emb_dim = segment_pos_emb1.shape[-1]
            return (
                torch.empty(0, C, device=segment_tokens.device),
                torch.empty(0, dtype=torch.long, device=segment_tokens.device),
                torch.empty(0, dtype=torch.long, device=segment_tokens.device),
                torch.empty(0, dtype=torch.long, device=segment_tokens.device),
                torch.empty(0, emb_dim, device=segment_tokens.device),
                torch.empty(0, emb_dim, device=segment_tokens.device),
            )

        # 提取静态token
        static_tokens = segment_tokens[:, static_mask, :]  # [T_seg, N_static, C]
        static_time_ids = segment_time_ids[:, static_mask]  # [T_seg, N_static]
        static_height_ids = segment_height_ids[:, static_mask]  # [T_seg, N_static]
        static_width_ids = segment_width_ids[:, static_mask]  # [T_seg, N_static]
        static_pos_emb1 = segment_pos_emb1[:, static_mask, :]  # [T_seg, N_static, emb_dim]
        static_pos_emb2 = segment_pos_emb2[:, static_mask, :]  # [T_seg, N_static, emb_dim]

        # 沿时间维度取平均
        merged_static = static_tokens.mean(dim=0)  # [N_static, C]
        merged_time_ids = static_time_ids.float().mean(dim=0).long()  # [N_static]
        merged_height_ids = static_height_ids.float().mean(dim=0).long()  # [N_static]
        merged_width_ids = static_width_ids.float().mean(dim=0).long()  # [N_static]
        merged_pos_emb1 = static_pos_emb1.mean(dim=0)  # [N_static, emb_dim]
        merged_pos_emb2 = static_pos_emb2.mean(dim=0)  # [N_static, emb_dim]

        return (
            merged_static,
            merged_time_ids,
            merged_height_ids,
            merged_width_ids,
            merged_pos_emb1,
            merged_pos_emb2,
        )

    def cluster_and_merge_dynamic_tokens(
        self,
        dynamic_tokens: torch.Tensor,
        dynamic_time_ids: torch.Tensor,
        dynamic_height_ids: torch.Tensor,
        dynamic_width_ids: torch.Tensor,
        dynamic_pos_emb1: torch.Tensor,
        dynamic_pos_emb2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对动态token进行空间聚类和合并

        Args:
            dynamic_tokens: [N_dynamic, C]
            dynamic_time_ids: [N_dynamic]
            dynamic_height_ids: [N_dynamic]
            dynamic_width_ids: [N_dynamic]
            dynamic_pos_emb1: [N_dynamic, emb_dim]
            dynamic_pos_emb2: [N_dynamic, emb_dim]

        Returns:
            merged_dynamic: [N_clustered, C]
            merged_time_ids: [N_clustered]
            merged_height_ids: [N_clustered]
            merged_width_ids: [N_clustered]
            merged_pos_emb1: [N_clustered, emb_dim]
            merged_pos_emb2: [N_clustered, emb_dim]
        """
        N_dynamic, C = dynamic_tokens.shape

        if N_dynamic <= 1:
            return (
                dynamic_tokens,
                dynamic_time_ids,
                dynamic_height_ids,
                dynamic_width_ids,
                dynamic_pos_emb1,
                dynamic_pos_emb2,
            )

        # 确定聚类数量
        num_clusters = max(1, int(N_dynamic * self.cluster_ratio))

        if num_clusters >= N_dynamic:
            # 不需要聚类
            return (
                dynamic_tokens,
                dynamic_time_ids,
                dynamic_height_ids,
                dynamic_width_ids,
                dynamic_pos_emb1,
                dynamic_pos_emb2,
            )

        # 使用DPC-KNN进行空间聚类
        cluster_indices, _ = cluster_dpc_knn(
            dynamic_tokens,
            cluster_num=num_clusters,
            k=min(self.dpc_knn_k, N_dynamic - 1),
            device=dynamic_tokens.device,
        )

        # 合并同一聚类的token
        merged_dynamic = merge_tokens_by_indices(
            dynamic_tokens, cluster_indices, num_clusters, method="mean"
        )

        # 合并位置信息（取平均）
        merged_time_ids = torch.zeros(num_clusters, dtype=torch.long, device=dynamic_tokens.device)
        merged_height_ids = torch.zeros(num_clusters, dtype=torch.long, device=dynamic_tokens.device)
        merged_width_ids = torch.zeros(num_clusters, dtype=torch.long, device=dynamic_tokens.device)
        merged_pos_emb1 = torch.zeros(num_clusters, dynamic_pos_emb1.shape[-1], device=dynamic_tokens.device)
        merged_pos_emb2 = torch.zeros(num_clusters, dynamic_pos_emb2.shape[-1], device=dynamic_tokens.device)

        for i in range(num_clusters):
            cluster_mask = (cluster_indices == i)
            if cluster_mask.sum() > 0:
                merged_time_ids[i] = dynamic_time_ids[cluster_mask].float().mean().long()
                merged_height_ids[i] = dynamic_height_ids[cluster_mask].float().mean().long()
                merged_width_ids[i] = dynamic_width_ids[cluster_mask].float().mean().long()
                merged_pos_emb1[i] = dynamic_pos_emb1[cluster_mask].mean(dim=0)
                merged_pos_emb2[i] = dynamic_pos_emb2[cluster_mask].mean(dim=0)

        return (
            merged_dynamic,
            merged_time_ids,
            merged_height_ids,
            merged_width_ids,
            merged_pos_emb1,
            merged_pos_emb2,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        video_grid_thw: List[List[int]],
        visual_token_indices: List[int],
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        执行时空token合并

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            position_ids: [3, batch_size, seq_len]
            position_embeddings: (pos_emb1, pos_emb2)
                - pos_emb1: [3, batch_size, seq_len, emb_dim]
                - pos_emb2: [3, batch_size, seq_len, emb_dim]
            video_grid_thw: [[t, h, w], ...]
            visual_token_indices: List of visual token positions
            batch_idx: 批次索引（默认0，假设batch_size=1）

        Returns:
            merged_hidden_states: [batch_size, new_seq_len, hidden_dim]
            merged_position_ids: [3, batch_size, new_seq_len]
            merged_position_embeddings: (merged_pos_emb1, merged_pos_emb2)
            stats: 统计信息
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # 提取该样本的数据
        sample_hidden = hidden_states[batch_idx]  # [seq_len, hidden_dim]
        sample_pos_ids = position_ids[:, batch_idx]  # [3, seq_len]
        sample_pos_emb1 = position_embeddings[0][:, batch_idx]  # [3, seq_len, emb_dim]
        sample_pos_emb2 = position_embeddings[1][:, batch_idx]  # [3, seq_len, emb_dim]

        # 提取位置信息
        time_ids = sample_pos_ids[0, :]  # [seq_len]
        height_ids = sample_pos_ids[1, :]  # [seq_len]
        width_ids = sample_pos_ids[2, :]  # [seq_len]

        # 提取视觉token
        visual_indices_set = set(visual_token_indices)
        visual_mask = torch.tensor([i in visual_indices_set for i in range(seq_len)], device=device)

        visual_hidden = sample_hidden[visual_mask]  # [num_visual, hidden_dim]
        visual_time_ids = time_ids[visual_mask]  # [num_visual]
        visual_height_ids = height_ids[visual_mask]  # [num_visual]
        visual_width_ids = width_ids[visual_mask]  # [num_visual]
        visual_pos_emb1 = sample_pos_emb1[:, visual_mask]  # [3, num_visual, emb_dim]
        visual_pos_emb2 = sample_pos_emb2[:, visual_mask]  # [3, num_visual, emb_dim]

        # 按帧分组
        unique_frames = visual_time_ids.unique().sort()[0]
        T = len(unique_frames)

        # 构建 [T, N, C] 的结构（假设每帧token数量相同）
        tokens_per_frame = []
        time_ids_per_frame = []
        height_ids_per_frame = []
        width_ids_per_frame = []
        pos_emb1_per_frame = []
        pos_emb2_per_frame = []

        for t in unique_frames:
            frame_mask = (visual_time_ids == t)
            tokens_per_frame.append(visual_hidden[frame_mask])
            time_ids_per_frame.append(visual_time_ids[frame_mask])
            height_ids_per_frame.append(visual_height_ids[frame_mask])
            width_ids_per_frame.append(visual_width_ids[frame_mask])
            pos_emb1_per_frame.append(visual_pos_emb1[:, frame_mask])
            pos_emb2_per_frame.append(visual_pos_emb2[:, frame_mask])

        # 检查每帧token数量是否相同
        frame_sizes = [t.shape[0] for t in tokens_per_frame]
        if len(set(frame_sizes)) != 1:
            if self.verbose:
                print(f"[Stage 1] Warning: Frame sizes not uniform: {frame_sizes}, skipping Stage 1")
            return hidden_states, position_ids, position_embeddings, {"enabled": False, "reason": "non_uniform_frames"}

        N = frame_sizes[0]
        frame_tokens = torch.stack(tokens_per_frame, dim=0)  # [T, N, C]
        frame_time_ids = torch.stack(time_ids_per_frame, dim=0)  # [T, N]
        frame_height_ids = torch.stack(height_ids_per_frame, dim=0)  # [T, N]
        frame_width_ids = torch.stack(width_ids_per_frame, dim=0)  # [T, N]
        frame_pos_emb1 = torch.stack([p.permute(1, 0, 2) for p in pos_emb1_per_frame], dim=0)  # [T, N, 3, emb_dim]
        frame_pos_emb2 = torch.stack([p.permute(1, 0, 2) for p in pos_emb2_per_frame], dim=0)  # [T, N, 3, emb_dim]

        # 1. 时序聚类
        cluster_indices, segments = self.temporal_clustering(frame_tokens, unique_frames)

        # 2. 对每个segment进行处理
        all_merged_tokens = []
        all_merged_time_ids = []
        all_merged_height_ids = []
        all_merged_width_ids = []
        all_merged_pos_emb1 = []
        all_merged_pos_emb2 = []

        tokens_before = T * N
        tokens_after = 0

        for seg_id, frame_ids in enumerate(segments):
            segment_tokens = frame_tokens[frame_ids]  # [T_seg, N, C]
            segment_time_ids = frame_time_ids[frame_ids]  # [T_seg, N]
            segment_height_ids = frame_height_ids[frame_ids]  # [T_seg, N]
            segment_width_ids = frame_width_ids[frame_ids]  # [T_seg, N]
            segment_pos_emb1 = frame_pos_emb1[frame_ids, :, 0, :]  # [T_seg, N, emb_dim] (只取时间通道)
            segment_pos_emb2 = frame_pos_emb2[frame_ids, :, 0, :]  # [T_seg, N, emb_dim]

            T_seg = len(frame_ids)

            # 2.1 静态/动态分离
            static_mask, _ = self.separate_static_dynamic(segment_tokens)
            dynamic_mask = ~static_mask

            # 2.2 合并静态token
            (
                merged_static,
                static_time_ids,
                static_height_ids,
                static_width_ids,
                static_pos_emb1,
                static_pos_emb2,
            ) = self.merge_static_tokens(
                segment_tokens,
                segment_time_ids,
                segment_height_ids,
                segment_width_ids,
                segment_pos_emb1,
                segment_pos_emb2,
                static_mask,
            )

            # 2.3 处理动态token（逐帧聚类）
            merged_dynamic_list = []
            dynamic_time_ids_list = []
            dynamic_height_ids_list = []
            dynamic_width_ids_list = []
            dynamic_pos_emb1_list = []
            dynamic_pos_emb2_list = []

            for i in range(T_seg):
                frame_dynamic_tokens = segment_tokens[i, dynamic_mask, :]  # [N_dynamic, C]
                frame_dynamic_time_ids = segment_time_ids[i, dynamic_mask]
                frame_dynamic_height_ids = segment_height_ids[i, dynamic_mask]
                frame_dynamic_width_ids = segment_width_ids[i, dynamic_mask]
                frame_dynamic_pos_emb1 = segment_pos_emb1[i, dynamic_mask, :]
                frame_dynamic_pos_emb2 = segment_pos_emb2[i, dynamic_mask, :]

                # 聚类该帧的动态token
                (
                    clustered_dynamic,
                    clustered_time_ids,
                    clustered_height_ids,
                    clustered_width_ids,
                    clustered_pos_emb1,
                    clustered_pos_emb2,
                ) = self.cluster_and_merge_dynamic_tokens(
                    frame_dynamic_tokens,
                    frame_dynamic_time_ids,
                    frame_dynamic_height_ids,
                    frame_dynamic_width_ids,
                    frame_dynamic_pos_emb1,
                    frame_dynamic_pos_emb2,
                )

                merged_dynamic_list.append(clustered_dynamic)
                dynamic_time_ids_list.append(clustered_time_ids)
                dynamic_height_ids_list.append(clustered_height_ids)
                dynamic_width_ids_list.append(clustered_width_ids)
                dynamic_pos_emb1_list.append(clustered_pos_emb1)
                dynamic_pos_emb2_list.append(clustered_pos_emb2)

            # 拼接动态token
            if len(merged_dynamic_list) > 0:
                merged_dynamic = torch.cat(merged_dynamic_list, dim=0)
                dynamic_time_ids = torch.cat(dynamic_time_ids_list, dim=0)
                dynamic_height_ids = torch.cat(dynamic_height_ids_list, dim=0)
                dynamic_width_ids = torch.cat(dynamic_width_ids_list, dim=0)
                dynamic_pos_emb1 = torch.cat(dynamic_pos_emb1_list, dim=0)
                dynamic_pos_emb2 = torch.cat(dynamic_pos_emb2_list, dim=0)
            else:
                emb_dim = segment_pos_emb1.shape[-1]
                merged_dynamic = torch.empty(0, hidden_dim, device=device)
                dynamic_time_ids = torch.empty(0, dtype=torch.long, device=device)
                dynamic_height_ids = torch.empty(0, dtype=torch.long, device=device)
                dynamic_width_ids = torch.empty(0, dtype=torch.long, device=device)
                dynamic_pos_emb1 = torch.empty(0, emb_dim, device=device)
                dynamic_pos_emb2 = torch.empty(0, emb_dim, device=device)

            # 2.4 拼接静态和动态token
            if merged_static.shape[0] > 0 and merged_dynamic.shape[0] > 0:
                segment_merged_tokens = torch.cat([merged_static, merged_dynamic], dim=0)
                segment_merged_time_ids = torch.cat([static_time_ids, dynamic_time_ids], dim=0)
                segment_merged_height_ids = torch.cat([static_height_ids, dynamic_height_ids], dim=0)
                segment_merged_width_ids = torch.cat([static_width_ids, dynamic_width_ids], dim=0)
                segment_merged_pos_emb1 = torch.cat([static_pos_emb1, dynamic_pos_emb1], dim=0)
                segment_merged_pos_emb2 = torch.cat([static_pos_emb2, dynamic_pos_emb2], dim=0)
            elif merged_static.shape[0] > 0:
                segment_merged_tokens = merged_static
                segment_merged_time_ids = static_time_ids
                segment_merged_height_ids = static_height_ids
                segment_merged_width_ids = static_width_ids
                segment_merged_pos_emb1 = static_pos_emb1
                segment_merged_pos_emb2 = static_pos_emb2
            else:
                segment_merged_tokens = merged_dynamic
                segment_merged_time_ids = dynamic_time_ids
                segment_merged_height_ids = dynamic_height_ids
                segment_merged_width_ids = dynamic_width_ids
                segment_merged_pos_emb1 = dynamic_pos_emb1
                segment_merged_pos_emb2 = dynamic_pos_emb2

            all_merged_tokens.append(segment_merged_tokens)
            all_merged_time_ids.append(segment_merged_time_ids)
            all_merged_height_ids.append(segment_merged_height_ids)
            all_merged_width_ids.append(segment_merged_width_ids)
            all_merged_pos_emb1.append(segment_merged_pos_emb1)
            all_merged_pos_emb2.append(segment_merged_pos_emb2)

            tokens_after += segment_merged_tokens.shape[0]

        # 3. 拼接所有segment
        merged_visual_tokens = torch.cat(all_merged_tokens, dim=0)
        merged_visual_time_ids = torch.cat(all_merged_time_ids, dim=0)
        merged_visual_height_ids = torch.cat(all_merged_height_ids, dim=0)
        merged_visual_width_ids = torch.cat(all_merged_width_ids, dim=0)
        merged_visual_pos_emb1 = torch.cat(all_merged_pos_emb1, dim=0)
        merged_visual_pos_emb2 = torch.cat(all_merged_pos_emb2, dim=0)

        # 4. 重建完整的序列（保留非视觉token）
        non_visual_mask = ~visual_mask
        non_visual_hidden = sample_hidden[non_visual_mask]
        non_visual_time_ids = time_ids[non_visual_mask]
        non_visual_height_ids = height_ids[non_visual_mask]
        non_visual_width_ids = width_ids[non_visual_mask]
        non_visual_pos_emb1 = sample_pos_emb1[:, non_visual_mask]
        non_visual_pos_emb2 = sample_pos_emb2[:, non_visual_mask]

        # 假设视觉token在序列开始位置（需要根据实际情况调整）
        # TODO: 更精确的合并策略，保持原有的文本token顺序
        new_hidden = torch.cat([merged_visual_tokens, non_visual_hidden], dim=0)
        new_time_ids = torch.cat([merged_visual_time_ids, non_visual_time_ids], dim=0)
        new_height_ids = torch.cat([merged_visual_height_ids, non_visual_height_ids], dim=0)
        new_width_ids = torch.cat([merged_visual_width_ids, non_visual_width_ids], dim=0)
        new_pos_emb1 = torch.cat([merged_visual_pos_emb1, non_visual_pos_emb1.permute(1, 0, 2)[:, 0, :]], dim=0)
        new_pos_emb2 = torch.cat([merged_visual_pos_emb2, non_visual_pos_emb2.permute(1, 0, 2)[:, 0, :]], dim=0)

        # 5. 重建batch格式
        # [batch_size, new_seq_len, hidden_dim]
        new_hidden_states = new_hidden.unsqueeze(0)

        # [3, batch_size, new_seq_len]
        new_position_ids = torch.stack([new_time_ids, new_height_ids, new_width_ids], dim=0).unsqueeze(1)

        # ([3, batch_size, new_seq_len, emb_dim], [3, batch_size, new_seq_len, emb_dim])
        # 注意：这里简化了，实际需要恢复完整的3通道
        new_pos_emb1_full = new_pos_emb1.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        new_pos_emb2_full = new_pos_emb2.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        new_position_embeddings = (new_pos_emb1_full, new_pos_emb2_full)

        # 统计信息
        stats = {
            "enabled": True,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_ratio": tokens_after / tokens_before if tokens_before > 0 else 1.0,
            "num_segments": len(segments),
        }

        if self.verbose:
            print(f"[Stage 1] Spatial-temporal merging:")
            print(f"  Tokens before: {tokens_before}")
            print(f"  Tokens after: {tokens_after}")
            print(f"  Reduction: {(1 - stats['reduction_ratio']) * 100:.1f}%")

        self.stats = stats

        return new_hidden_states, new_position_ids, new_position_embeddings, stats


def create_spatial_temporal_token_merger(
    tau: float = 0.8,
    cluster_ratio: float = 0.5,
    temporal_segment_ratio: float = 0.25,
    dpc_knn_k: int = 5,
    verbose: bool = False,
) -> SpatialTemporalTokenMerger:
    """
    创建时空token合并器

    Args:
        tau: 静态/动态分离阈值
        cluster_ratio: 空间聚类保留比例
        temporal_segment_ratio: 时序分段比例
        dpc_knn_k: DPC-KNN的k近邻参数
        verbose: 是否打印详细信息

    Returns:
        merger: SpatialTemporalTokenMerger实例
    """
    return SpatialTemporalTokenMerger(
        tau=tau,
        cluster_ratio=cluster_ratio,
        temporal_segment_ratio=temporal_segment_ratio,
        dpc_knn_k=dpc_knn_k,
        verbose=verbose,
    )
