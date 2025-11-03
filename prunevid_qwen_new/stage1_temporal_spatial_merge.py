"""
Stage 1: 时空Token合并
====================

实现PruneVid的第一阶段：通过时序和空间维度的token合并来减少视频冗余。

核心流程（参考论文图1和图2）：
1. 时序聚类：将视频帧聚类成场景段（Temporal Clustering）
2. 静态/动态分离：识别每个场景中的静态区域和动态区域
3. 静态token时序合并：对静态区域在时间维度上取平均
4. 空间聚类和合并：对静态和动态token分别进行空间聚类和合并

论文公式参考：
- 公式(2)：余弦相似度 s_i^(t,t') = X_v^(t)(i)^T X_v^(t')(i) / (||X_v^(t)(i)|| ||X_v^(t')(i)||)
- 公式(3)：平均相似度 s̄_i = 2/(|T_b|(|T_b|-1)) Σ s_i^(t,t')
- 公式(4)：静态token合并 X̃_v^(b)(i) = 1/|T_b| Σ X_v^(t)(i)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from utils import (
    temporal_clustering_continuous,
    detect_static_tokens,
    dpc_knn_clustering,
    merge_tokens_by_labels,
    average_position_embeddings,
    average_position_ids,
    compute_compression_stats,
)
from config import PruneVidConfig


class SpatialTemporalTokenMerger(nn.Module):
    """
    时空Token合并器

    这个类实现了PruneVid的Stage 1，负责减少视频的时空冗余。
    """

    def __init__(self, config: PruneVidConfig):
        """
        初始化

        Args:
            config: PruneVid配置对象
        """
        super().__init__()
        self.config = config
        self.tau = config.tau
        self.cluster_ratio = config.cluster_ratio
        self.temporal_segment_ratio = config.temporal_segment_ratio
        self.dpc_knn_k = config.dpc_knn_k
        self.verbose = config.verbose

        # 统计信息
        self.last_stats = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_token_start: int,
        visual_token_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        执行时空token合并

        Args:
            hidden_states: [batch, seq_len, hidden_dim] 输入序列
            position_ids: [3, batch, seq_len] 3D位置ID（时间、高度、宽度）
            position_embeddings: Tuple of (cos, sin)，shape [3, batch, seq_len, head_dim]
            visual_token_start: 视觉token开始位置
            visual_token_end: 视觉token结束位置

        Returns:
            merged_hidden_states: [batch, new_seq_len, hidden_dim] 合并后的序列
            merged_position_ids: [3, batch, new_seq_len] 更新后的位置ID
            merged_position_embeddings: Tuple of (cos, sin) 更新后的位置embedding
            stats: 统计信息字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # 提取视觉token
        visual_tokens = hidden_states[:, visual_token_start:visual_token_end, :]  # [batch, num_visual, hidden_dim]
        num_visual_tokens = visual_tokens.shape[1]

        if self.verbose:
            print(f"[Stage 1] 输入视觉token数量: {num_visual_tokens}")

        # 提取视觉部分的position_ids
        visual_position_ids = position_ids[:, :, visual_token_start:visual_token_end]  # [3, batch, num_visual]

        # 提取position_embeddings（如果有的话）
        cos, sin = position_embeddings
        visual_cos = cos[:, :, visual_token_start:visual_token_end, :]  # [3, batch, num_visual, head_dim]
        visual_sin = sin[:, :, visual_token_start:visual_token_end, :]

        # 处理batch - 循环处理每个batch样本
        batch_results_hidden = []
        batch_results_pos_ids = []
        batch_results_cos = []
        batch_results_sin = []
        batch_stats = []

        for batch_idx in range(batch_size):
            # 提取当前batch的数据
            # [batch, num_visual, hidden_dim] -> [num_visual, hidden_dim]
            visual_tokens_single = visual_tokens[batch_idx]
            visual_position_ids_single = visual_position_ids[:, batch_idx, :]  # [3, num_visual]

            # 步骤1：根据position_ids将token重组为帧格式
            # position_ids[0]是时间维度，用它来分组
            time_ids = visual_position_ids_single[0]  # [num_visual]
            unique_times = time_ids.unique().sort()[0]  # 排序后的唯一时间戳
            T = len(unique_times)  # 帧数

            if self.verbose:
                print(f"[Stage 1 Batch {batch_idx}] 检测到 {T} 帧")

            # 构建 [T, N, hidden_dim] 的张量，N可能每帧不同
            # 先找出每帧的token数量
            tokens_per_frame = []
            indices_per_frame = []
            for t in unique_times:
                mask = (time_ids == t)
                frame_tokens = visual_tokens_single[mask]  # [N_t, hidden_dim]
                frame_indices = torch.where(mask)[0]
                tokens_per_frame.append(frame_tokens)
                indices_per_frame.append(frame_indices)
    
            # 检查每帧token数量是否一致
            num_tokens_per_frame = [len(tokens) for tokens in tokens_per_frame]
            if len(set(num_tokens_per_frame)) != 1:
                # 不一致，可能是padding导致的，取最小值
                min_tokens = min(num_tokens_per_frame)
                if self.verbose:
                    print(f"[Stage 1] 警告：每帧token数量不一致: {num_tokens_per_frame}，将截断到 {min_tokens}")
                tokens_per_frame = [tokens[:min_tokens] for tokens in tokens_per_frame]
                indices_per_frame = [indices[:min_tokens] for indices in indices_per_frame]
                N = min_tokens
            else:
                N = num_tokens_per_frame[0]
    
            # 堆叠成 [T, N, hidden_dim]
            frame_tokens = torch.stack(tokens_per_frame, dim=0)
    
            if self.verbose:
                print(f"[Stage 1] 每帧token数量: {N}")
                print(f"[Stage 1] 视频token shape: {frame_tokens.shape}")
    
            # 步骤2：时序聚类 - 将帧聚类成场景段
            # 计算每帧的平均特征
            frame_features = frame_tokens.mean(dim=1)  # [T, hidden_dim]
            segments = temporal_clustering_continuous(
                frame_features,
                k=self.dpc_knn_k,
                ratio=self.temporal_segment_ratio
            )
    
            num_segments = len(segments)
            if self.verbose:
                print(f"[Stage 1] 时序聚类得到 {num_segments} 个场景段")
                for i, seg in enumerate(segments):
                    print(f"  段 {i}: 帧 {seg[0]} 到 {seg[-1]} (共{len(seg)}帧)")
    
            # 步骤3：对每个场景段处理
            merged_tokens_list = []
            merged_position_ids_list = []
            merged_cos_list = []
            merged_sin_list = []
    
            total_static_tokens = 0
            total_dynamic_tokens = 0
    
            for seg_idx, seg_frames in enumerate(segments):
                seg_T = len(seg_frames)
                # 提取该段的token [seg_T, N, hidden_dim]
                seg_tokens = frame_tokens[seg_frames]
    
                if self.verbose:
                    print(f"\n[Stage 1] 处理场景段 {seg_idx} (帧 {seg_frames})")
    
                # 3.1 静态/动态分离
                static_mask, dynamic_mask = detect_static_tokens(
                    seg_tokens, seg_frames, tau=self.tau
                )
    
                num_static = static_mask.sum().item()
                num_dynamic = dynamic_mask.sum().item()
                total_static_tokens += num_static
                total_dynamic_tokens += num_dynamic
    
                if self.verbose:
                    print(f"  静态token: {num_static}/{N} ({num_static/N*100:.1f}%)")
                    print(f"  动态token: {num_dynamic}/{N} ({num_dynamic/N*100:.1f}%)")
    
                # 3.2 处理静态token：时序合并
                if num_static > 0:
                    static_tokens = seg_tokens[:, static_mask, :]  # [seg_T, num_static, hidden_dim]
                    # 时序平均（论文公式4）
                    merged_static = static_tokens.mean(dim=0)  # [num_static, hidden_dim]
    
                    # 空间聚类和合并
                    if num_static > 1:
                        static_labels, _ = dpc_knn_clustering(
                            merged_static,
                            k=self.dpc_knn_k,
                            ratio=self.cluster_ratio
                        )
                        # 按聚类标签合并
                        final_static = merge_tokens_by_labels(merged_static, static_labels)
                    else:
                        final_static = merged_static  # 只有1个，不需要聚类
    
                    # 位置信息：取该段第一帧的静态token位置
                    first_frame_idx = seg_frames[0]
                    first_frame_indices = indices_per_frame[first_frame_idx]
                    static_indices = first_frame_indices[static_mask]
    
                    # 提取position_ids和embeddings
                    static_pos_ids = visual_position_ids_single[:, static_indices]  # [3, num_static]
                    static_pos_cos = visual_cos[:, batch_idx, static_indices, :]  # [3, num_static, head_dim]
                    static_pos_sin = visual_sin[:, batch_idx, static_indices, :]
    
                    # 如果进行了聚类，需要平均位置信息
                    if num_static > 1:
                        static_pos_ids = average_position_ids(static_pos_ids, static_labels)
                        static_pos_cos, static_pos_sin = average_position_embeddings(
                            (static_pos_cos, static_pos_sin), static_labels
                        )
    
                    merged_tokens_list.append(final_static)
                    merged_position_ids_list.append(static_pos_ids)
                    merged_cos_list.append(static_pos_cos)
                    merged_sin_list.append(static_pos_sin)
    
                    if self.verbose:
                        print(f"  静态token合并: {num_static} -> {final_static.shape[0]}")
    
                # 3.3 处理动态token：只做空间聚类（不做时序合并）
                if num_dynamic > 0:
                    # 对每帧的动态token分别处理
                    for frame_idx_in_seg in range(seg_T):
                        frame_dynamic_tokens = seg_tokens[frame_idx_in_seg, dynamic_mask, :]  # [num_dynamic, hidden_dim]
    
                        # 空间聚类
                        if num_dynamic > 1:
                            dynamic_labels, _ = dpc_knn_clustering(
                                frame_dynamic_tokens,
                                k=self.dpc_knn_k,
                                ratio=self.cluster_ratio
                            )
                            final_dynamic = merge_tokens_by_labels(frame_dynamic_tokens, dynamic_labels)
                        else:
                            final_dynamic = frame_dynamic_tokens
                            dynamic_labels = torch.zeros(1, dtype=torch.long, device=device)
    
                        # 位置信息
                        actual_frame_idx = seg_frames[frame_idx_in_seg]
                        frame_indices = indices_per_frame[actual_frame_idx]
                        dynamic_indices = frame_indices[dynamic_mask]
    
                        dynamic_pos_ids = visual_position_ids_single[:, dynamic_indices]
                        dynamic_pos_cos = visual_cos[:, batch_idx, dynamic_indices, :]
                        dynamic_pos_sin = visual_sin[:, batch_idx, dynamic_indices, :]
    
                        if num_dynamic > 1:
                            dynamic_pos_ids = average_position_ids(dynamic_pos_ids, dynamic_labels)
                            dynamic_pos_cos, dynamic_pos_sin = average_position_embeddings(
                                (dynamic_pos_cos, dynamic_pos_sin), dynamic_labels
                            )
    
                        merged_tokens_list.append(final_dynamic)
                        merged_position_ids_list.append(dynamic_pos_ids)
                        merged_cos_list.append(dynamic_pos_cos)
                        merged_sin_list.append(dynamic_pos_sin)
    
                    if self.verbose:
                        print(f"  动态token处理: {num_dynamic} -> {final_dynamic.shape[0]} (每帧)")
    
            # 步骤4：合并所有段的结果
            if len(merged_tokens_list) > 0:
                all_merged_tokens = torch.cat(merged_tokens_list, dim=0)  # [total_merged, hidden_dim]
                all_merged_pos_ids = torch.cat(merged_position_ids_list, dim=1)  # [3, total_merged]
                all_merged_cos = torch.cat(merged_cos_list, dim=1)  # [3, total_merged, head_dim]
                all_merged_sin = torch.cat(merged_sin_list, dim=1)
            else:
                # 没有token（不应该发生）
                raise RuntimeError("合并后没有token！")
    
            num_merged = all_merged_tokens.shape[0]

            if self.verbose:
                print(f"\n[Stage 1 Batch {batch_idx}] 最终合并结果: {num_visual_tokens} -> {num_merged} tokens")
                print(f"  压缩率: {num_merged/num_visual_tokens*100:.1f}%")
                print(f"  减少率: {(1-num_merged/num_visual_tokens)*100:.1f}%")

            # 收集当前batch的结果
            batch_results_hidden.append(all_merged_tokens)  # [num_merged, hidden_dim]
            batch_results_pos_ids.append(all_merged_pos_ids)  # [3, num_merged]
            batch_results_cos.append(all_merged_cos)  # [3, num_merged, head_dim]
            batch_results_sin.append(all_merged_sin)  # [3, num_merged, head_dim]

            # 收集统计信息
            batch_stats.append({
                "num_segments": num_segments,
                "num_static_tokens": total_static_tokens,
                "num_dynamic_tokens": total_dynamic_tokens,
                "static_ratio": total_static_tokens / (total_static_tokens + total_dynamic_tokens),
                "original_tokens": num_visual_tokens,
                "merged_tokens": num_merged,
            })

        # 步骤5：合并所有batch的结果并重建完整序列
        # 检查所有batch的合并后token数量是否相同
        merged_lengths = [t.shape[0] for t in batch_results_hidden]
        if len(set(merged_lengths)) != 1:
            # 不同batch合并后长度不同，需要padding（暂不支持）
            raise NotImplementedError(
                f"不同batch样本合并后的token数量不同: {merged_lengths}。"
                f"当前实现要求batch内所有样本的视频结构相同。"
            )

        # 堆叠所有batch的结果
        # batch_results_hidden: List[[num_merged, hidden_dim]] -> [batch, num_merged, hidden_dim]
        all_merged_tokens = torch.stack(batch_results_hidden, dim=0)

        # batch_results_pos_ids: List[[3, num_merged]] -> [3, batch, num_merged]
        all_merged_pos_ids = torch.stack(batch_results_pos_ids, dim=1)

        # batch_results_cos/sin: List[[3, num_merged, head_dim]] -> [3, batch, num_merged, head_dim]
        all_merged_cos = torch.stack(batch_results_cos, dim=1)
        all_merged_sin = torch.stack(batch_results_sin, dim=1)

        # 重建完整序列（文本token + 合并后的视觉token）
        # [batch, seq_len, hidden_dim] -> [文本前] + [视觉] + [文本后]
        text_before = hidden_states[:, :visual_token_start, :]
        text_after = hidden_states[:, visual_token_end:, :]

        new_hidden_states = torch.cat([text_before, all_merged_tokens, text_after], dim=1)
        new_seq_len = new_hidden_states.shape[1]

        # 重建position_ids
        pos_ids_before = position_ids[:, :, :visual_token_start]
        pos_ids_after = position_ids[:, :, visual_token_end:]

        new_position_ids = torch.cat([pos_ids_before, all_merged_pos_ids, pos_ids_after], dim=2)

        # 重建position_embeddings
        cos_before = cos[:, :, :visual_token_start, :]
        cos_after = cos[:, :, visual_token_end:, :]
        sin_before = sin[:, :, :visual_token_start, :]
        sin_after = sin[:, :, visual_token_end:, :]

        new_cos = torch.cat([cos_before, all_merged_cos, cos_after], dim=2)
        new_sin = torch.cat([sin_before, all_merged_sin, sin_after], dim=2)

        new_position_embeddings = (new_cos, new_sin)

        # 聚合统计信息
        if batch_size > 0:
            avg_stats = {
                "num_segments": sum(s["num_segments"] for s in batch_stats) / batch_size,
                "num_static_tokens": sum(s["num_static_tokens"] for s in batch_stats) / batch_size,
                "num_dynamic_tokens": sum(s["num_dynamic_tokens"] for s in batch_stats) / batch_size,
                "static_ratio": sum(s["static_ratio"] for s in batch_stats) / batch_size,
            }
            stats = compute_compression_stats(
                sum(s["original_tokens"] for s in batch_stats),
                sum(s["merged_tokens"] for s in batch_stats)
            )
            stats.update(avg_stats)
        else:
            stats = {}

        self.last_stats = stats

        return new_hidden_states, new_position_ids, new_position_embeddings, stats


# 导出
__all__ = ["SpatialTemporalTokenMerger"]
