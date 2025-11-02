"""
Qwen2.5-VL Stage 1 Wrapper for PruneVid
Spatial-Temporal Token Merging at Vision Encoder Level

This wrapper hooks into Qwen2.5-VL's vision encoder to apply Stage 1 token merging
before the visual tokens enter the LLM.
"""

import torch
import torch.nn.functional as F
from typing import Any, Tuple, List
from einops import einsum

from .stage1_utils import (
    cluster_dpc_knn,
    refine_clusters,
    segment_lengths,
    compute_cluster_vectors
)


class Qwen25VLStage1Wrapper:
    """
    Wrapper for Qwen2.5-VL to apply Stage 1 Spatial-Temporal Token Merging.

    Hooks into the model's get_video_features() method to apply token merging
    after vision encoder but before entering LLM.

    Args:
        model: Qwen2_5_VLForConditionalGeneration instance
        tau: Threshold for static/dynamic region detection (0.7-0.9)
        cluster_ratio: Ratio of spatial tokens to keep after clustering (0.3-0.8)
        temporal_segment_ratio: Ratio for temporal segmentation (0.25-0.5)
        enable_stage1: Whether to enable Stage 1 processing
        verbose: Print debug information
    """

    def __init__(
        self,
        model: Any,
        tau: float = 0.8,
        cluster_ratio: float = 0.5,
        temporal_segment_ratio: float = 0.25,
        enable_stage1: bool = True,
        verbose: bool = False
    ):
        self.model = model
        self.tau = tau
        self.cluster_ratio = cluster_ratio
        self.temporal_segment_ratio = temporal_segment_ratio
        self.enable_stage1 = enable_stage1
        self.verbose = verbose

        # Statistics
        self.stats = {
            'tokens_before_stage1': 0,
            'tokens_after_stage1': 0,
            'stage1_compression_ratio': 0.0,
            'num_static_tokens': 0,
            'num_dynamic_tokens': 0,
        }

        # Store original method
        self.original_get_video_features = None

        if self.enable_stage1:
            self._hook_get_video_features()

            if self.verbose:
                print(f"\n[Stage 1 Wrapper] Initialized:")
                print(f"  tau: {self.tau}")
                print(f"  cluster_ratio: {self.cluster_ratio}")
                print(f"  temporal_segment_ratio: {self.temporal_segment_ratio}")

    def _hook_get_video_features(self):
        """Hook the model's get_video_features method to apply Stage 1."""
        if not hasattr(self.model.model, 'get_video_features'):
            raise AttributeError(
                "Model does not have 'get_video_features' method. "
                "This wrapper only works with Qwen2.5-VL models."
            )

        # Store original method
        self.original_get_video_features = self.model.model.get_video_features

        # Define wrapped method
        def wrapped_get_video_features(pixel_values_videos, video_grid_thw):
            # Call original method
            video_embeds_list = self.original_get_video_features(
                pixel_values_videos, video_grid_thw
            )

            # Apply Stage 1 to each video
            merged_embeds_list = []
            for i, embeds in enumerate(video_embeds_list):
                grid_thw = video_grid_thw[i:i+1]  # [1, 3]
                merged = self.apply_stage1(embeds, grid_thw)
                merged_embeds_list.append(merged)

            return merged_embeds_list

        # Replace with wrapped method
        self.model.model.get_video_features = wrapped_get_video_features

        if self.verbose:
            print(f"[Stage 1 Wrapper] Hooked get_video_features method")

    def apply_stage1(
        self,
        video_embeds: torch.Tensor,
        grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Stage 1: Spatial-Temporal Token Merging.

        Args:
            video_embeds: [num_tokens, hidden_dim] - flat visual tokens
            grid_thw: [1, 3] = [t, h, w] - temporal/spatial grid info

        Returns:
            merged_embeds: [num_merged_tokens, hidden_dim] - compressed tokens
        """
        t, h, w = grid_thw[0].tolist()
        num_tokens, hidden_dim = video_embeds.shape

        # Record original token count
        self.stats['tokens_before_stage1'] = num_tokens

        # CRITICAL FIX: Qwen2.5-VL has built-in PatchMerger that reduces spatial tokens
        # Calculate actual spatial tokens per frame from total tokens
        actual_spatial_per_frame = num_tokens // t

        if self.verbose:
            print(f"\n[Stage 1] Processing video:")
            print(f"  Grid (t, h, w): ({t}, {h}, {w}) [from image_grid_thw]")
            print(f"  Expected tokens (without PatchMerger): {t * h * w}")
            print(f"  Actual tokens: {num_tokens}")
            print(f"  Actual spatial tokens per frame: {actual_spatial_per_frame}")
            if actual_spatial_per_frame != h * w:
                compression_ratio = (h * w) / actual_spatial_per_frame
                print(f"  ⚠️  PatchMerger detected: {compression_ratio:.1f}x spatial compression")

        # Reshape: [num_tokens, hidden_dim] -> [1, t, actual_spatial, hidden_dim]
        # Use actual spatial tokens, not grid h*w (which is pre-PatchMerger)
        video_embeds_reshaped = video_embeds.view(1, t, actual_spatial_per_frame, hidden_dim)

        # Apply merge_frames_dynamic
        merged_embeds, static_sizes, dynamic_sizes, window_sizes = self.merge_frames_dynamic(
            video_embeds_reshaped,
            num_frames=t,
            num_spatial=actual_spatial_per_frame,  # Use actual spatial tokens
            threshold=self.tau,
            k=7
        )

        # Update statistics
        self.stats['tokens_after_stage1'] = merged_embeds.shape[1]
        self.stats['stage1_compression_ratio'] = 1.0 - (
            self.stats['tokens_after_stage1'] / self.stats['tokens_before_stage1']
        )
        self.stats['num_static_tokens'] = sum(static_sizes)
        self.stats['num_dynamic_tokens'] = sum(dynamic_sizes)

        if self.verbose:
            print(f"  After Stage 1: {self.stats['tokens_after_stage1']} tokens")
            print(f"  Compression: {self.stats['stage1_compression_ratio']:.1%}")
            print(f"  Static tokens: {self.stats['num_static_tokens']}")
            print(f"  Dynamic tokens: {self.stats['num_dynamic_tokens']}")

        # Reshape back to flat format: [1, num_merged_tokens, hidden_dim] -> [num_merged_tokens, hidden_dim]
        return merged_embeds.squeeze(0)

    def merge_frames_dynamic(
        self,
        frames: torch.Tensor,
        num_frames: int,
        num_spatial: int,
        threshold: float = 0.8,
        k: int = 7
    ) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
        """
        Dynamic frame merging with static/dynamic region separation.

        Algorithm:
            1. Temporal Segmentation: cluster frames into temporal segments
            2. For each segment:
                a. Calculate frame similarity to detect static regions
                b. Static regions: temporal averaging + spatial clustering
                c. Dynamic regions: per-frame spatial clustering
            3. Concatenate all processed tokens

        Args:
            frames: [B, T, L, C] where T=num_frames, L=num_spatial_tokens
            num_frames: number of video frames
            num_spatial: number of spatial tokens per frame
            threshold: similarity threshold for static/dynamic (tau parameter)
            k: number of nearest neighbors for clustering

        Returns:
            final_features: [B, num_merged_tokens, C] merged tokens
            static_sizes: list of static token counts per segment
            dynamic_sizes: list of dynamic token counts per segment
            window_sizes: list of temporal segment lengths
        """
        B, T, L, C = frames.shape

        # Step 1: Temporal Segmentation
        # Cluster frames based on temporal similarity
        num_segments = max(1, int(num_frames * self.temporal_segment_ratio))

        # Use mean spatial feature for temporal clustering
        frame_features = frames.mean(dim=2)  # [B, T, C]
        idx_clusters, _ = cluster_dpc_knn(frame_features, cluster_num=num_segments, k=k)
        idx_clusters = refine_clusters(idx_clusters)
        window_list = segment_lengths(idx_clusters)  # [B, num_segments]

        static_features = []
        dynamic_features = []
        static_sizes = []
        dynamic_sizes = []

        # Step 2: Process each temporal segment
        start_idx = 0
        for window_size in window_list[0]:  # Assume batch size = 1
            if window_size == 0:
                break

            window_size = window_size.item()

            # Get frames for current segment
            current_frames = frames[:, start_idx:start_idx+window_size, :, :]  # [B, W, L, C]

            # Step 2a: Calculate frame similarity within segment
            # Normalize features
            frames_normed = F.normalize(current_frames, p=2, dim=-1)

            # Compute pairwise frame similarity for each spatial location
            # [B, W, L, C] × [B, W, L, C] -> [B, W, W, L] -> [B, L]
            frames_sim = einsum(frames_normed, frames_normed, 'b w l c, b t l c -> b w t l')

            # Average similarity across frame pairs for each spatial location
            if window_size > 1:
                frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size * (window_size - 1))
            else:
                frames_sim = frames_sim.squeeze(1).squeeze(1)  # [B, L]

            # Step 2b: Create static/dynamic mask
            mask = frames_sim > threshold  # [B, L] - True = static
            mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C)  # [B, W, L, C]

            # Step 2c: Process static regions
            # Static: temporal averaging (collapse temporal dimension)
            static_mask = mask_expand
            if static_mask.sum() > 0:
                static_feat = torch.masked_select(current_frames, static_mask).view(B, window_size, -1, C).mean(dim=1)

                # Apply spatial clustering if enough tokens
                if static_feat.shape[1] > 14:
                    num_cluster = max(1, int(static_feat.shape[1] * self.cluster_ratio))
                    static_feat = self.spatial_merge_tokens(static_feat, num_cluster=num_cluster, k=7)
            else:
                static_feat = torch.zeros(B, 0, C, device=frames.device, dtype=frames.dtype)

            static_features.append(static_feat)
            static_sizes.append(static_feat.shape[1])

            # Step 2d: Process dynamic regions
            # Dynamic: keep per-frame, apply spatial clustering to each
            dynamic_mask = ~mask_expand
            if dynamic_mask.sum() > 0:
                dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, window_size, -1, C)

                dynamic_window_list = []
                for i in range(window_size):
                    dynamic_feat_frame = dynamic_feat[:, i, :, :]  # [B, num_dynamic_spatial, C]

                    # Apply spatial clustering if enough tokens
                    if dynamic_feat_frame.shape[1] > 14:
                        num_cluster = max(1, int(dynamic_feat_frame.shape[1] * self.cluster_ratio))
                        dynamic_feat_frame = self.spatial_merge_tokens(dynamic_feat_frame, num_cluster=num_cluster, k=7)

                    dynamic_window_list.append(dynamic_feat_frame)

                dynamic_feat = torch.cat(dynamic_window_list, dim=1)  # [B, total_dynamic, C]
            else:
                dynamic_feat = torch.zeros(B, 0, C, device=frames.device, dtype=frames.dtype)

            dynamic_features.append(dynamic_feat)
            dynamic_sizes.append(dynamic_feat.shape[1])

            start_idx += window_size

        # Step 3: Concatenate all features (static and dynamic interleaved)
        final_features = []
        for static_feature, dynamic_feature in zip(static_features, dynamic_features):
            final_features.append(static_feature)
            final_features.append(dynamic_feature)
        final_features = torch.cat(final_features, dim=1)  # [B, total_tokens, C]

        window_sizes = window_list[0].tolist()
        # Remove padding zeros
        window_sizes = [w for w in window_sizes if w > 0]

        return final_features, static_sizes, dynamic_sizes, window_sizes

    def spatial_merge_tokens(
        self,
        features: torch.Tensor,
        num_cluster: int,
        k: int = 7
    ) -> torch.Tensor:
        """
        Apply spatial clustering to merge tokens.

        Args:
            features: [B, L, C] feature vectors
            num_cluster: number of clusters to form
            k: number of nearest neighbors for DPC-KNN

        Returns:
            cluster_features: [B, num_cluster, C] merged features
        """
        cluster_idx, _ = cluster_dpc_knn(features, cluster_num=num_cluster, k=k)
        cluster_features = compute_cluster_vectors(features, cluster_idx, num_cluster=num_cluster)
        return cluster_features

    def get_stats(self) -> dict:
        """Get Stage 1 statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics for new generation."""
        self.stats = {
            'tokens_before_stage1': 0,
            'tokens_after_stage1': 0,
            'stage1_compression_ratio': 0.0,
            'num_static_tokens': 0,
            'num_dynamic_tokens': 0,
        }

    def unhook(self):
        """Restore original get_video_features method."""
        if self.original_get_video_features is not None:
            self.model.model.get_video_features = self.original_get_video_features
            self.original_get_video_features = None

            if self.verbose:
                print(f"[Stage 1 Wrapper] Unhooked get_video_features method")
