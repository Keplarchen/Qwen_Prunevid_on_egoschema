"""
PruneVid工具函数
===============

包含DPC-KNN聚类、相似度计算、token合并等核心工具函数。

主要功能：
1. DPC-KNN聚类算法（Du et al., 2016）
2. 余弦相似度计算
3. 静态token检测
4. Token合并和位置ID管理
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


def cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    计算特征矩阵的余弦相似度矩阵

    Args:
        features: [N, C] 特征矩阵

    Returns:
        similarity: [N, N] 余弦相似度矩阵
    """
    # L2归一化
    features_norm = F.normalize(features, p=2, dim=1)
    # 计算余弦相似度
    similarity = torch.mm(features_norm, features_norm.t())
    return similarity


def compute_pairwise_cosine_similarity(tokens_seq: List[torch.Tensor]) -> torch.Tensor:
    """
    计算时序token序列的成对余弦相似度

    用于检测静态token：如果某个空间位置在时间维度上的token相似度很高，
    说明该位置是静态的（背景或静止物体）

    Args:
        tokens_seq: List of [C] tensors，长度为T（时间步数）

    Returns:
        avg_similarity: 标量，平均余弦相似度
    """
    T = len(tokens_seq)
    if T <= 1:
        return torch.tensor(1.0, device=tokens_seq[0].device)

    # Stack成 [T, C]
    tokens = torch.stack(tokens_seq, dim=0)  # [T, C]

    # 计算相似度矩阵 [T, T]
    sim_matrix = cosine_similarity_matrix(tokens)

    # 只取上三角（不包括对角线），避免重复计算
    # 平均相似度 = 上三角和 / (T*(T-1)/2)
    upper_triangle = torch.triu(sim_matrix, diagonal=1)
    avg_sim = upper_triangle.sum() / (T * (T - 1) / 2)

    return avg_sim


def dpc_knn_clustering(
    features: torch.Tensor,
    k: int = 5,
    ratio: float = 0.25,
    return_centers: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    DPC-KNN聚类算法（Density Peaks Clustering with k-Nearest Neighbors）

    参考论文：Du et al., "Study on density peaks clustering based on
              k-nearest neighbors and principal component analysis", 2016

    算法步骤：
    1. 计算每个点的局部密度（基于K近邻距离）
    2. 计算每个点到更高密度点的最小距离
    3. 计算决策值 = 密度 × 距离
    4. 选择top ratio的点作为聚类中心
    5. 将其他点分配到最近的聚类中心

    Args:
        features: [N, C] 特征矩阵
        k: K近邻数量
        ratio: 聚类中心比例（0-1之间）
        return_centers: 是否返回聚类中心的索引

    Returns:
        labels: [N] 聚类标签，从0开始
        centers: Optional[Tensor] 如果return_centers=True，返回聚类中心的索引
    """
    N, C = features.shape
    device = features.device

    if N <= 1:
        labels = torch.zeros(N, dtype=torch.long, device=device)
        centers = torch.arange(N, device=device) if return_centers else None
        return labels, centers

    # 1. 计算距离矩阵 [N, N]
    dist_matrix = torch.cdist(features, features, p=2)

    # 2. 计算局部密度 ρ_i = 1 / (平均k近邻距离 + eps)
    k_actual = min(k + 1, N)  # +1因为包括自己
    k_nearest_dists, _ = torch.topk(dist_matrix, k_actual, largest=False, dim=1)
    # k_nearest_dists: [N, k+1]，第一列是0（自己到自己）
    avg_k_dist = k_nearest_dists[:, 1:].mean(dim=1)  # [N]，排除自己
    rho = 1.0 / (avg_k_dist + 1e-8)  # 局部密度

    # 3. 计算δ_i = 到更高密度点的最小距离
    delta = torch.zeros(N, device=device)
    for i in range(N):
        # 找到密度比i高的点
        higher_density_mask = rho > rho[i]
        if higher_density_mask.any():
            # 到这些点的距离
            dists_to_higher = dist_matrix[i, higher_density_mask]
            delta[i] = dists_to_higher.min()
        else:
            # 如果是密度最大的点，设为到其他所有点的最大距离
            delta[i] = dist_matrix[i].max()

    # 4. 计算决策值 γ_i = ρ_i × δ_i
    gamma = rho * delta

    # 5. 选择聚类中心：取top ratio的点
    num_centers = max(1, int(N * ratio))
    _, center_indices = torch.topk(gamma, num_centers, largest=True)

    # 6. 为所有点分配聚类标签
    labels = torch.zeros(N, dtype=torch.long, device=device)

    # 先标记聚类中心
    for cluster_id, center_idx in enumerate(center_indices):
        labels[center_idx] = cluster_id

    # 为非中心点分配标签：分配到最近的聚类中心
    non_center_mask = torch.ones(N, dtype=torch.bool, device=device)
    non_center_mask[center_indices] = False
    non_center_indices = torch.where(non_center_mask)[0]

    if len(non_center_indices) > 0:
        # 计算非中心点到所有聚类中心的距离
        dists_to_centers = dist_matrix[non_center_indices][:, center_indices]  # [非中心数, 中心数]
        # 分配到最近的中心
        nearest_center = dists_to_centers.argmin(dim=1)  # [非中心数]
        labels[non_center_indices] = nearest_center

    if return_centers:
        return labels, center_indices
    else:
        return labels, None


def temporal_clustering_continuous(
    frame_features: torch.Tensor,
    k: int = 5,
    ratio: float = 0.25
) -> List[List[int]]:
    """
    对视频帧进行时序聚类，并确保聚类结果是连续的

    这个函数专门用于视频场景分割，确保同一场景的帧是连续的。
    如果聚类算法产生了不连续的聚类，会进行后处理修正。

    Args:
        frame_features: [T, C] 每帧的特征（平均池化后）
        k: DPC-KNN的k值
        ratio: 聚类中心比例

    Returns:
        segments: List[List[int]]，每个元素是一个场景的帧索引列表
                  例如：[[0,1,2,3], [4,5,6], [7,8,9,10,11]]
    """
    T = frame_features.shape[0]

    if T <= 1:
        return [[i] for i in range(T)]

    # 1. DPC-KNN聚类
    labels, _ = dpc_knn_clustering(frame_features, k=k, ratio=ratio)
    labels = labels.cpu().numpy()

    # 2. 后处理：确保连续性
    # 策略：扫描帧序列，如果某帧的标签与前一帧不同，检查是否应该属于前一个聚类
    corrected_labels = labels.copy()

    for i in range(1, T):
        if labels[i] != labels[i-1]:
            # 检查这个标签的变化是否合理
            # 如果i+1也是不同的标签，说明i是孤立的，应该归并到邻近聚类
            if i < T - 1 and labels[i] != labels[i+1]:
                # i是孤立点，归并到前一个聚类
                corrected_labels[i] = corrected_labels[i-1]

    # 3. 构建连续的段
    segments = []
    current_segment = [0]
    current_label = corrected_labels[0]

    for i in range(1, T):
        if corrected_labels[i] == current_label:
            current_segment.append(i)
        else:
            # 新的聚类开始
            segments.append(current_segment)
            current_segment = [i]
            current_label = corrected_labels[i]

    # 添加最后一个段
    segments.append(current_segment)

    return segments


def detect_static_tokens(
    frame_tokens: torch.Tensor,
    frame_indices: List[int],
    tau: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    检测静态token和动态token

    对于每个空间位置，计算其在时间维度上的平均余弦相似度。
    如果相似度 >= tau，认为是静态token。

    Args:
        frame_tokens: [T, N, C] 某个时序段的token
                      T=帧数，N=每帧token数，C=特征维度
        frame_indices: List[int] 该段包含的帧索引（用于调试）
        tau: 静态检测阈值

    Returns:
        static_mask: [N] bool tensor，True表示静态
        dynamic_mask: [N] bool tensor，True表示动态
    """
    T, N, C = frame_tokens.shape
    device = frame_tokens.device

    if T <= 1:
        # 只有一帧，全部视为动态
        static_mask = torch.zeros(N, dtype=torch.bool, device=device)
        dynamic_mask = torch.ones(N, dtype=torch.bool, device=device)
        return static_mask, dynamic_mask

    # 对每个空间位置计算平均相似度
    avg_similarities = torch.zeros(N, device=device)

    for i in range(N):
        # 取出位置i在所有帧的token [T, C]
        tokens_at_i = frame_tokens[:, i, :]
        # 计算相似度矩阵 [T, T]
        sim_matrix = cosine_similarity_matrix(tokens_at_i)
        # 平均值（排除对角线）
        mask = ~torch.eye(T, dtype=torch.bool, device=device)
        avg_similarities[i] = sim_matrix[mask].mean()

    # 根据阈值分类
    static_mask = avg_similarities >= tau
    dynamic_mask = ~static_mask

    return static_mask, dynamic_mask


def merge_tokens_by_labels(
    tokens: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    根据聚类标签合并token（每个聚类内取平均）

    Args:
        tokens: [N, C] token矩阵
        labels: [N] 聚类标签

    Returns:
        merged_tokens: [K, C] 合并后的token，K=聚类数
    """
    unique_labels = labels.unique().sort()[0]
    K = len(unique_labels)
    C = tokens.shape[1]
    device = tokens.device

    merged_tokens = torch.zeros(K, C, device=device, dtype=tokens.dtype)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        merged_tokens[i] = tokens[mask].mean(dim=0)

    return merged_tokens


def average_position_embeddings(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据聚类标签平均Rotary Position Embeddings

    Qwen2.5-VL使用rotary embeddings，格式为(cos, sin)的tuple

    Args:
        position_embeddings: Tuple of (cos, sin)
                            cos/sin shape: [3, batch, seq_len, head_dim]
                            (3对应时间、高度、宽度三个维度)
        labels: [seq_len] 聚类标签

    Returns:
        merged_embeddings: Tuple of (cos, sin)，shape: [3, batch, K, head_dim]
    """
    cos, sin = position_embeddings
    # cos/sin: [3, batch, seq_len, head_dim]

    unique_labels = labels.unique().sort()[0]
    K = len(unique_labels)

    # 初始化合并后的embedding
    merged_cos = torch.zeros(
        cos.shape[0], cos.shape[1], K, cos.shape[3],
        device=cos.device, dtype=cos.dtype
    )
    merged_sin = torch.zeros_like(merged_cos)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        # 平均cos和sin
        merged_cos[:, :, i, :] = cos[:, :, mask, :].mean(dim=2)
        merged_sin[:, :, i, :] = sin[:, :, mask, :].mean(dim=2)

    return (merged_cos, merged_sin)


def average_position_ids(
    position_ids: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    根据聚类标签平均Position IDs

    Qwen2.5-VL的position_ids是3D的：[3, batch, seq_len]
    分别对应时间、高度、宽度

    Args:
        position_ids: [3, batch, seq_len]
        labels: [seq_len] 聚类标签

    Returns:
        merged_position_ids: [3, batch, K]
    """
    unique_labels = labels.unique().sort()[0]
    K = len(unique_labels)
    device = position_ids.device

    merged_ids = torch.zeros(
        3, position_ids.shape[1], K,
        device=device, dtype=position_ids.dtype
    )

    for i, label in enumerate(unique_labels):
        mask = labels == label
        # 对每个维度分别平均（然后取整）
        for dim in range(3):
            merged_ids[dim, :, i] = position_ids[dim, :, mask].float().mean(dim=1).round().long()

    return merged_ids


def compute_compression_stats(
    original_num_tokens: int,
    compressed_num_tokens: int
) -> Dict[str, float]:
    """
    计算压缩统计信息

    Args:
        original_num_tokens: 原始token数量
        compressed_num_tokens: 压缩后token数量

    Returns:
        stats: 包含各种统计指标的字典
    """
    retention_ratio = compressed_num_tokens / original_num_tokens
    reduction_ratio = 1 - retention_ratio
    compression_ratio = original_num_tokens / compressed_num_tokens

    return {
        "original_tokens": original_num_tokens,
        "compressed_tokens": compressed_num_tokens,
        "retention_ratio": retention_ratio,
        "reduction_ratio": reduction_ratio,
        "compression_ratio": compression_ratio,
        "reduction_percentage": reduction_ratio * 100,
    }


# 测试函数（可选）
def test_dpc_knn():
    """测试DPC-KNN聚类"""
    print("Testing DPC-KNN clustering...")

    # 生成测试数据：3个聚类
    torch.manual_seed(42)
    cluster1 = torch.randn(10, 5) + torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    cluster2 = torch.randn(10, 5) + torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    cluster3 = torch.randn(10, 5) + torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0])

    features = torch.cat([cluster1, cluster2, cluster3], dim=0)

    labels, centers = dpc_knn_clustering(features, k=5, ratio=0.1, return_centers=True)

    print(f"Total points: {len(features)}")
    print(f"Number of clusters: {labels.unique().numel()}")
    print(f"Cluster centers: {centers}")
    print(f"Labels: {labels}")
    print("DPC-KNN test passed!")


if __name__ == "__main__":
    test_dpc_knn()
