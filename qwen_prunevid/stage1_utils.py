"""
Stage 1 Utility Functions for PruneVid
Spatial-Temporal Token Merging utilities migrated from PLLaVA

These functions implement the DPC-KNN clustering algorithm and related utilities
for temporal segmentation and spatial token merging.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Sample features following the index.

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Returns:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(
    x: torch.Tensor,
    cluster_num: int,
    k: int = 5,
    token_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, int]:
    """
    Cluster tokens with DPC-KNN algorithm.

    Algorithm:
        1. Compute pairwise distance matrix
        2. Calculate local density using k-nearest neighbors
        3. Calculate distance indicator (distance to nearest higher-density point)
        4. Score = density × distance, select top cluster_num as cluster centers
        5. Assign each token to nearest cluster center

    Args:
        x: input token feature, [B, N, C]
        cluster_num (int): number of clusters
        k (int): number of nearest neighbors for local density calculation
        token_mask (Tensor[B, N]): mask indicating valid tokens (non-zero = valid)

    Returns:
        idx_cluster (Tensor[B, N]): cluster index of each token
        cluster_num (int): actual cluster number (same as input)
    """
    with torch.no_grad():
        B, N, C = x.shape

        # Compute pairwise distance matrix, normalized by sqrt(C)
        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # Set distance between empty tokens and any other tokens to max
            # to not affect local density calculation
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # Step 1: Get local density
        # Use k-nearest neighbors' distances
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

        # Add small noise to ensure no tokens have identical density
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # Empty tokens should have density = 0
            density = density * token_mask

        # Step 2: Get distance indicator
        # For each token, find distance to nearest token with higher density
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # Step 3: Select clustering centers according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # Step 4: Assign tokens to nearest center
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # Step 5: Make sure cluster centers merge to themselves
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def segment_lengths(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the length of each contiguous segment in a cluster index tensor.

    For example:
        tensor = [[0, 0, 1, 1, 1, 0, 0]]
        returns [[2, 3, 2]] (lengths of segments)

    Args:
        tensor: cluster index tensor, [B, N]

    Returns:
        result: segment lengths, [B, max_segments]
                padded with 0s if fewer than max_segments
    """
    device = tensor.device
    B, N = tensor.shape

    segment_lengths_list = []
    max_segments = 0

    for i in range(B):
        seq = tensor[i]
        # Find positions where value changes
        change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
        # Include start and end positions
        boundaries = torch.cat([
            torch.tensor([0], device=device),
            change_points,
            torch.tensor([N], device=device)
        ])
        # Calculate length of each segment
        lengths = boundaries[1:] - boundaries[:-1]
        segment_lengths_list.append(lengths)
        max_segments = max(max_segments, lengths.numel())

    # Initialize result tensor with zeros
    result = torch.zeros((B, max_segments), dtype=torch.long, device=device)

    # Fill in segment lengths for each sample
    for i in range(B):
        lengths = segment_lengths_list[i]
        result[i, :lengths.numel()] = lengths

    return result


def refine_clusters(cluster_idx: torch.Tensor) -> torch.Tensor:
    """
    Refine clustering results by removing fragmented segments.

    For each cluster:
        1. Find all contiguous segments belonging to that cluster
        2. Keep only the longest segment
        3. Re-assign shorter segments to neighboring clusters

    This ensures temporal coherence: each cluster forms one contiguous block,
    avoiding fragmentation across the temporal dimension.

    Args:
        cluster_idx: Tensor of shape (B, N), cluster indices

    Returns:
        refined_cluster_idx: Tensor of shape (B, N), refined cluster indices
    """
    B, N = cluster_idx.shape
    refined_cluster_idx = cluster_idx.clone()

    for b in range(B):
        clusters = torch.unique(cluster_idx[b])
        segment_info = {}

        # Step 1: For each cluster, find all contiguous segments
        for cluster_label in clusters:
            indices = (cluster_idx[b] == cluster_label).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            # Find contiguous segments
            segments = []
            start = indices[0].item()
            prev = indices[0].item()
            for idx in indices[1:]:
                idx = idx.item()
                if idx == prev + 1:
                    prev = idx
                else:
                    # New segment
                    segments.append((start, prev))
                    start = idx
                    prev = idx
            # Add last segment
            segments.append((start, prev))
            segment_info[cluster_label.item()] = segments

        # Step 2: Keep only the longest segment for each cluster
        for cluster_label, segments in segment_info.items():
            # Find longest segment length
            max_length = 0
            for (start, end) in segments:
                length = end - start + 1
                if length > max_length:
                    max_length = length

            # If longest segment has length 1, remove this cluster
            if max_length == 1:
                for (start, end) in segments:
                    refined_cluster_idx[b, start:end+1] = -1  # -1 = needs reassignment
                continue

            # Keep longest segment, mark others for reassignment
            for (start, end) in segments:
                length = end - start + 1
                if length == max_length:
                    continue  # Keep this segment
                else:
                    refined_cluster_idx[b, start:end+1] = -1  # Needs reassignment

        # Step 3: Reassign fragments to neighboring clusters
        # Choose the neighbor with the longer contiguous segment
        idx = 0
        while idx < N:
            if refined_cluster_idx[b, idx] == -1:
                # Find fragment that needs reassignment
                start = idx
                while idx < N and refined_cluster_idx[b, idx] == -1:
                    idx += 1
                end = idx - 1

                # Find left and right neighbor clusters and their segment lengths
                left_cluster_label = None
                left_length = 0
                if start > 0:
                    left_label = refined_cluster_idx[b, start - 1].item()
                    # Calculate left segment length
                    l_idx = start - 1
                    while l_idx >= 0 and refined_cluster_idx[b, l_idx] == left_label:
                        l_idx -= 1
                    left_length = start - l_idx - 1
                    left_cluster_label = left_label

                right_cluster_label = None
                right_length = 0
                if end < N - 1:
                    right_label = refined_cluster_idx[b, end + 1].item()
                    # Calculate right segment length
                    r_idx = end + 1
                    while r_idx < N and refined_cluster_idx[b, r_idx] == right_label:
                        r_idx += 1
                    right_length = r_idx - end - 1
                    right_cluster_label = right_label

                # Choose neighbor with longer segment (prefer left if tie)
                if left_length > right_length:
                    new_label = left_cluster_label
                elif right_length > left_length:
                    new_label = right_cluster_label
                else:
                    new_label = left_cluster_label if left_cluster_label is not None else right_cluster_label

                # If no neighbors exist, default to cluster 0
                if new_label is None:
                    new_label = 0

                # Reassign fragment
                refined_cluster_idx[b, start:end+1] = new_label
            else:
                idx += 1

    return refined_cluster_idx


def compute_cluster_vectors(
    features: torch.Tensor,
    cluster_idx: torch.Tensor,
    num_cluster: int
) -> torch.Tensor:
    """
    Compute average feature vector for each cluster.

    Uses efficient one-hot encoding + matrix multiplication approach.

    Args:
        features: feature vectors, [B, L, D]
        cluster_idx: cluster indices for each vector, [B, L]
        num_cluster: total number of clusters

    Returns:
        cluster_features: averaged features for each cluster, [B, num_cluster, D]
    """
    B, L, D = features.shape

    # Step 1: One-hot encode cluster indices
    # cluster_idx_onehot: [B, L, num_cluster]
    cluster_idx_onehot = F.one_hot(cluster_idx, num_classes=num_cluster).to(dtype=features.dtype)

    # Step 2: Compute feature sum for each cluster
    # Transpose to [B, num_cluster, L]
    cluster_idx_onehot_t = cluster_idx_onehot.permute(0, 2, 1)

    # Matrix multiplication: [B, num_cluster, L] × [B, L, D] = [B, num_cluster, D]
    cluster_sums = torch.bmm(cluster_idx_onehot_t, features)

    # Step 3: Compute count of elements in each cluster
    # cluster_counts: [B, num_cluster]
    cluster_counts = cluster_idx_onehot.sum(dim=1)

    # Step 4: Compute average features
    # Avoid division by zero
    cluster_counts_nonzero = cluster_counts.clone()
    cluster_counts_nonzero[cluster_counts_nonzero == 0] = 1

    # Calculate average: [B, num_cluster, D]
    cluster_features = cluster_sums / cluster_counts_nonzero.unsqueeze(-1)

    # Step 5: Set features of empty clusters to 0
    zero_mask = (cluster_counts == 0).unsqueeze(-1)  # [B, num_cluster, 1]
    cluster_features = cluster_features.masked_fill(zero_mask, 0)

    return cluster_features
