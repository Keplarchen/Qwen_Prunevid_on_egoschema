"""
PruneVid工具函数

包含DPC-KNN聚类、相似度计算等工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


def cluster_dpc_knn(
    features: torch.Tensor,
    cluster_num: int,
    k: int = 5,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    密度峰值聚类算法 (Density Peaks Clustering with k-Nearest Neighbors)

    论文参考：
    "Study on density peaks clustering based on k-nearest neighbors and principal component analysis"
    (Du et al., 2016)

    Args:
        features: 特征张量，shape [N, C]
        cluster_num: 目标聚类数量
        k: k近邻数量
        device: 计算设备

    Returns:
        idx_clusters: 每个样本的聚类索引，shape [N]
        centers: 聚类中心的索引，shape [cluster_num]
    """
    N, C = features.shape
    features = features.to(device)

    # 1. 计算距离矩阵 [N, N]
    # 注意：cdist在CUDA上不支持BFloat16，需要转换为Float32
    original_dtype = features.dtype
    if original_dtype == torch.bfloat16:
        features_for_cdist = features.float()
    else:
        features_for_cdist = features

    dist_matrix = torch.cdist(features_for_cdist, features_for_cdist, p=2)

    # 2. 计算局部密度 rho（基于k近邻）
    # 对每个点，找到k个最近邻居，计算平均距离的倒数
    topk_dist, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    # topk_dist[:, 0]是自己到自己的距离（0），取[1:k+1]
    rho = 1.0 / (topk_dist[:, 1:k + 1].mean(dim=1) + 1e-8)

    # 3. 计算决策距离 delta
    # delta[i] = min_{j: rho[j] > rho[i]} dist[i, j]
    # 如果rho[i]是最大的，则delta[i] = max(dist[i, :])
    delta = torch.zeros(N, device=device)
    rho_sorted_idx = torch.argsort(rho, descending=True)

    # 对于密度最大的点
    delta[rho_sorted_idx[0]] = dist_matrix[rho_sorted_idx[0]].max()

    # 对于其他点
    for i in range(1, N):
        idx = rho_sorted_idx[i]
        # 找到所有密度比当前点高的点
        higher_density_indices = rho_sorted_idx[:i]
        # 计算到这些点的最小距离
        delta[idx] = dist_matrix[idx, higher_density_indices].min()

    # 4. 计算gamma = rho * delta（选择聚类中心的指标）
    gamma = rho * delta

    # 5. 选择gamma最大的cluster_num个点作为聚类中心
    _, centers = torch.topk(gamma, cluster_num, largest=True)
    centers = centers.sort()[0]  # 排序以保持顺序

    # 6. 将其他点分配到最近的聚类中心
    # 按照密度从高到低的顺序分配
    idx_clusters = torch.zeros(N, dtype=torch.long, device=device)
    idx_clusters[centers] = torch.arange(cluster_num, device=device)

    for i in range(N):
        if i in rho_sorted_idx[:cluster_num]:
            continue
        idx = rho_sorted_idx[i]
        # 找到所有已分配的点
        assigned_mask = torch.zeros(N, dtype=torch.bool, device=device)
        assigned_mask[rho_sorted_idx[:i]] = True
        # 找到最近的已分配点
        dist_to_assigned = dist_matrix[idx].clone()
        dist_to_assigned[~assigned_mask] = float('inf')
        nearest_assigned = dist_to_assigned.argmin()
        idx_clusters[idx] = idx_clusters[nearest_assigned]

    return idx_clusters, centers


def compute_cosine_similarity(
    tokens1: torch.Tensor,
    tokens2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    计算两组token之间的余弦相似度

    Args:
        tokens1: [N, C] 或 [B, N, C]
        tokens2: [M, C] 或 [B, M, C]
        normalize: 是否归一化

    Returns:
        similarity: [N, M] 或 [B, N, M]
    """
    if normalize:
        tokens1 = F.normalize(tokens1, p=2, dim=-1)
        tokens2 = F.normalize(tokens2, p=2, dim=-1)

    if tokens1.dim() == 2:
        similarity = torch.mm(tokens1, tokens2.T)
    else:
        similarity = torch.bmm(tokens1, tokens2.transpose(1, 2))

    return similarity


def merge_tokens_by_indices(
    tokens: torch.Tensor,
    cluster_indices: torch.Tensor,
    num_clusters: int,
    method: str = "mean"
) -> torch.Tensor:
    """
    根据聚类索引合并token

    Args:
        tokens: [N, C] 原始token
        cluster_indices: [N] 每个token的聚类索引
        num_clusters: 聚类数量
        method: 合并方法，'mean' 或 'max'

    Returns:
        merged_tokens: [num_clusters, C] 合并后的token
    """
    N, C = tokens.shape
    device = tokens.device

    merged_tokens = torch.zeros(num_clusters, C, device=device, dtype=tokens.dtype)

    for i in range(num_clusters):
        mask = cluster_indices == i
        if mask.sum() == 0:
            continue

        cluster_tokens = tokens[mask]
        if method == "mean":
            merged_tokens[i] = cluster_tokens.mean(dim=0)
        elif method == "max":
            merged_tokens[i] = cluster_tokens.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown merge method: {method}")

    return merged_tokens


def temporal_similarity_matrix(
    frame_tokens: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    计算视频帧之间的相似度矩阵

    Args:
        frame_tokens: [T, N, C] T帧，每帧N个token
        normalize: 是否归一化

    Returns:
        similarity: [T, T] 帧间相似度矩阵
    """
    T, N, C = frame_tokens.shape

    # 对每一帧计算平均特征
    frame_features = frame_tokens.mean(dim=1)  # [T, C]

    # 计算相似度
    similarity = compute_cosine_similarity(frame_features, frame_features, normalize)

    return similarity


def detect_static_tokens(
    frame_tokens: torch.Tensor,
    threshold: float = 0.8,
    method: str = "pairwise_mean"
) -> torch.Tensor:
    """
    检测静态token（在时间维度上变化很小的token）

    Args:
        frame_tokens: [T, N, C] T帧，每帧N个token
        threshold: 相似度阈值
        method: 检测方法
            - 'pairwise_mean': 计算所有帧对之间的平均相似度
            - 'consecutive': 仅计算相邻帧之间的相似度

    Returns:
        static_mask: [N] 布尔mask，True表示静态token
    """
    T, N, C = frame_tokens.shape
    device = frame_tokens.device

    if method == "pairwise_mean":
        # 对每个空间位置，计算所有帧对之间的平均相似度
        static_scores = torch.zeros(N, device=device)

        for i in range(N):
            token_seq = frame_tokens[:, i, :]  # [T, C]
            # 计算相似度矩阵 [T, T]
            sim_matrix = compute_cosine_similarity(token_seq, token_seq, normalize=True)
            # 取上三角（不包括对角线）的平均值
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            if mask.sum() > 0:
                static_scores[i] = sim_matrix[mask].mean()

    elif method == "consecutive":
        # 仅计算相邻帧之间的相似度
        static_scores = torch.zeros(N, device=device)

        for i in range(N):
            token_seq = frame_tokens[:, i, :]  # [T, C]
            # 计算相邻帧之间的相似度
            similarities = []
            for t in range(T - 1):
                sim = F.cosine_similarity(
                    token_seq[t].unsqueeze(0),
                    token_seq[t + 1].unsqueeze(0),
                    dim=1
                )
                similarities.append(sim)
            static_scores[i] = torch.stack(similarities).mean()

    else:
        raise ValueError(f"Unknown method: {method}")

    static_mask = static_scores >= threshold

    return static_mask


def enforce_temporal_continuity(
    cluster_indices: torch.Tensor,
    timestamps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    强制时序连续性约束：确保连续的帧被分配到连续的聚类

    用于DPC-KNN聚类后，确保同一场景的连续帧在同一聚类中

    Args:
        cluster_indices: [T] 每帧的聚类索引
        timestamps: [T] 可选，每帧的时间戳

    Returns:
        adjusted_indices: [T] 调整后的聚类索引
    """
    T = len(cluster_indices)
    adjusted_indices = cluster_indices.clone()

    # 简单策略：如果一个帧的聚类与前后帧都不同，则分配给前一帧的聚类
    for t in range(1, T - 1):
        if adjusted_indices[t] != adjusted_indices[t - 1] and \
           adjusted_indices[t] != adjusted_indices[t + 1] and \
           adjusted_indices[t - 1] == adjusted_indices[t + 1]:
            adjusted_indices[t] = adjusted_indices[t - 1]

    return adjusted_indices


def compute_token_importance_from_attention(
    attention_weights: torch.Tensor,
    aggregation: str = "max"
) -> torch.Tensor:
    """
    从注意力权重计算token重要性

    Args:
        attention_weights: [B, num_heads, seq_len_q, seq_len_k] 注意力权重
        aggregation: 聚合方法
            - 'max': 取最大值（论文使用）
            - 'mean': 取平均值

    Returns:
        importance: [B, seq_len_k] 每个token的重要性分数
    """
    B, num_heads, seq_len_q, seq_len_k = attention_weights.shape

    if aggregation == "max":
        # 先对query维度取max，再对head维度取max
        importance = attention_weights.max(dim=2)[0].max(dim=1)[0]  # [B, seq_len_k]
    elif aggregation == "mean":
        # 对query和head维度取平均
        importance = attention_weights.mean(dim=[1, 2])  # [B, seq_len_k]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return importance


def select_top_k_tokens(
    importance_scores: torch.Tensor,
    keep_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    选择重要性最高的top-k个token

    Args:
        importance_scores: [B, N] 或 [N] 重要性分数
        keep_ratio: 保留比例

    Returns:
        selected_indices: 选中的token索引
        pruned_indices: 被剪枝的token索引
    """
    if importance_scores.dim() == 1:
        N = len(importance_scores)
        k = max(1, int(N * keep_ratio))

        # 选择top-k
        _, selected_indices = torch.topk(importance_scores, k, largest=True, sorted=True)
        # 剩余的为被剪枝的
        all_indices = torch.arange(N, device=importance_scores.device)
        mask = torch.ones(N, dtype=torch.bool, device=importance_scores.device)
        mask[selected_indices] = False
        pruned_indices = all_indices[mask]

    else:
        # Batch处理
        B, N = importance_scores.shape
        k = max(1, int(N * keep_ratio))

        _, selected_indices = torch.topk(importance_scores, k, dim=1, largest=True, sorted=True)
        # 对于batch，返回每个样本的索引
        pruned_indices = None  # 暂不支持batch的pruned_indices

    return selected_indices, pruned_indices


def video_frame_sampling(
    num_frames_total: int,
    target_frames: int,
    method: str = "uniform"
) -> List[int]:
    """
    视频帧采样

    Args:
        num_frames_total: 视频总帧数
        target_frames: 目标采样帧数
        method: 采样方法
            - 'uniform': 均匀采样
            - 'center': 中心采样

    Returns:
        frame_indices: 采样的帧索引列表
    """
    if num_frames_total <= target_frames:
        return list(range(num_frames_total))

    if method == "uniform":
        indices = np.linspace(0, num_frames_total - 1, target_frames, dtype=int)
    elif method == "center":
        start = (num_frames_total - target_frames) // 2
        indices = np.arange(start, start + target_frames)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return indices.tolist()
