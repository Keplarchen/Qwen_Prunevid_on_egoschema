"""
Qwen2.5-VL Stage 1 Wrapper V2 for PruneVid
Spatial-Temporal Token Merging in LLM Model Forward (after validation)

This version hooks into the LLM model's forward method (after embeddings are created)
instead of the vision encoder, avoiding token count mismatch issues.

Key differences from V1:
- V1: Hook get_video_features() → compress before validation → fails
- V2: Hook model.forward() → compress after validation → success
"""

import torch
import torch.nn.functional as F
from typing import Any, Tuple, List, Optional
from einops import einsum

from .stage1_utils import (
    cluster_dpc_knn,
    refine_clusters,
    segment_lengths,
    compute_cluster_vectors
)


class Qwen25VLStage1WrapperV2:
    """
    Wrapper for Qwen2.5-VL to apply Stage 1 Spatial-Temporal Token Merging.

    This version hooks into Qwen2_5_VLModel.forward() to apply token merging
    AFTER embeddings are created and validation passes.

    Architecture:
        1. Vision Encoder → full video embeddings (e.g., 3136 tokens)
        2. Insert to inputs_embeds → validation passes ✓
        3. [NEW] LLM Model forward → Stage 1 merging (3136 → 880 tokens)
        4. LLM Processing → uses compressed tokens

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

        # For Stage 2 compatibility: track actual visual token positions after compression
        self.compressed_visual_positions = None  # (visual_start, visual_end) after Stage 1

        # Store original method
        self.original_model_forward = None

        # Cache for tracking video info during forward
        self.current_video_grid_thw = None
        self.current_pixel_values_videos = None

        if self.enable_stage1:
            self._hook_model_forward()

            if self.verbose:
                print(f"\n[Stage 1 Wrapper V2] Initialized:")
                print(f"  tau: {self.tau}")
                print(f"  cluster_ratio: {self.cluster_ratio}")
                print(f"  temporal_segment_ratio: {self.temporal_segment_ratio}")
                print(f"  Hook: LLM model.forward (after validation)")

    def _hook_model_forward(self):
        """Hook the LLM model's forward method to apply Stage 1 after validation."""
        # Hook Qwen2_5_VLModel (the LLM model, not ForConditionalGeneration)
        if not hasattr(self.model, 'model'):
            raise AttributeError(
                "Model does not have 'model' attribute. "
                "This wrapper requires Qwen2_5_VLForConditionalGeneration."
            )

        llm_model = self.model.model  # Qwen2_5_VLModel
        self.original_model_forward = llm_model.forward

        # Define wrapped method
        def wrapped_model_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            **kwargs
        ):
            """
            Wrapped forward that applies Stage 1 token merging.

            At this point:
            - inputs_embeds already contains visual embeddings (validation passed)
            - We can safely modify inputs_embeds and position_ids
            """

            # CRITICAL FIX: In generate(), inputs_embeds is None on first call!
            # We need to create embeddings first if input_ids is provided but inputs_embeds is None
            original_input_ids = input_ids  # Save for Stage 1 processing
            if inputs_embeds is None and input_ids is not None:
                # Get the embedding layer (Qwen uses get_input_embeddings())
                embed_layer = llm_model.get_input_embeddings()
                inputs_embeds = embed_layer(input_ids)

            # Apply Stage 1 if we have video inputs and this is the prefill stage
            apply_stage1 = (
                self.enable_stage1
                and inputs_embeds is not None
                and original_input_ids is not None
                and self.current_video_grid_thw is not None
                and (cache_position is None or cache_position[0] == 0)  # Prefill stage only
            )

            # Debug: Show why apply_stage1 is True/False
            if self.verbose:
                print(f"\n[Stage 1 V2 Debug] Forward called:")
                print(f"  enable_stage1: {self.enable_stage1}")
                print(f"  inputs_embeds is not None: {inputs_embeds is not None}")
                print(f"  original_input_ids is not None: {original_input_ids is not None}")
                print(f"  current_video_grid_thw is not None: {self.current_video_grid_thw is not None}")
                if self.current_video_grid_thw is not None:
                    print(f"    → video_grid_thw: {self.current_video_grid_thw}")
                print(f"  cache_position: {cache_position}")
                if cache_position is not None:
                    print(f"    → cache_position[0]: {cache_position[0]}")
                print(f"  → apply_stage1: {apply_stage1}")

            if apply_stage1:
                if self.verbose:
                    print(f"\n[Stage 1 V2] Applying token merging in LLM forward")
                    print(f"  inputs_embeds shape: {inputs_embeds.shape}")
                    if attention_mask is not None:
                        print(f"  attention_mask shape: {attention_mask.shape}")

                # Apply Stage 1 token merging
                inputs_embeds, position_ids = self._apply_stage1_to_embeddings(
                    inputs_embeds=inputs_embeds,
                    input_ids=original_input_ids,
                    position_ids=position_ids,
                    video_grid_thw=self.current_video_grid_thw
                )

                # CRITICAL: Compress all sequence-length-dependent tensors
                new_seq_len = inputs_embeds.shape[1]

                # Compress attention_mask
                if attention_mask is not None:
                    attention_mask = self._compress_attention_mask(
                        attention_mask=attention_mask,
                        input_ids=original_input_ids,
                        new_seq_len=new_seq_len
                    )

                # Compress cache_position (critical for causal_mask generation)
                if cache_position is not None:
                    cache_position = self._compress_cache_position(
                        cache_position=cache_position,
                        input_ids=original_input_ids,
                        new_seq_len=new_seq_len
                    )

                if self.verbose:
                    print(f"  After Stage 1: {inputs_embeds.shape}")
                    if attention_mask is not None:
                        print(f"    attention_mask: {attention_mask.shape}")
                    if cache_position is not None:
                        print(f"    cache_position: {cache_position.shape}")

            # After processing (with or without Stage 1), set input_ids=None to use inputs_embeds
            if inputs_embeds is not None:
                input_ids = None

            # If Stage 1 was applied, remove video-related kwargs to prevent validation errors
            # Qwen's get_placeholder_mask would check token count vs feature count and fail
            if apply_stage1:
                # Filter out video/image grid parameters that trigger validation
                filtered_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ['image_grid_thw', 'video_grid_thw', 'pixel_values',
                                  'pixel_values_videos', 'image_embeds', 'video_embeds']
                }
                if self.verbose and kwargs:
                    removed_keys = set(kwargs.keys()) - set(filtered_kwargs.keys())
                    if removed_keys:
                        print(f"  Removed kwargs to prevent validation: {removed_keys}")
            else:
                filtered_kwargs = kwargs

            # Call original forward with potentially modified inputs_embeds
            return self.original_model_forward(
                input_ids=input_ids,  # None if Stage 1 applied, original otherwise
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **filtered_kwargs
            )

        # Replace with wrapped method
        llm_model.forward = wrapped_model_forward

        if self.verbose:
            print(f"[Stage 1 Wrapper V2] Hooked LLM model.forward")

    def _apply_stage1_to_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        video_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply Stage 1 token merging to inputs_embeds.

        Args:
            inputs_embeds: [batch_size, seq_len, hidden_dim] - embeddings with visual tokens
            input_ids: [batch_size, seq_len] - for locating visual tokens
            position_ids: [3, batch_size, seq_len] or None - M-ROPE position IDs
            video_grid_thw: [num_videos, 3] - (t, h, w) for each video

        Returns:
            compressed_embeds: [batch_size, new_seq_len, hidden_dim]
            compressed_position_ids: [3, batch_size, new_seq_len] or None
        """
        batch_size, seq_len, hidden_dim = inputs_embeds.shape

        # Get video token ID from config
        video_token_id = self.model.config.video_token_id

        # Process each sample in the batch
        compressed_embeds_list = []
        compressed_pos_ids_list = []

        for i in range(batch_size):
            sample_embeds = inputs_embeds[i]  # [seq_len, hidden_dim]
            sample_input_ids = input_ids[i]  # [seq_len]

            # Find video token positions
            video_token_mask = (sample_input_ids == video_token_id)
            video_token_indices = video_token_mask.nonzero(as_tuple=True)[0]

            if len(video_token_indices) == 0:
                # No video tokens, keep as is
                compressed_embeds_list.append(sample_embeds.unsqueeze(0))
                if position_ids is not None:
                    compressed_pos_ids_list.append(position_ids[:, i:i+1, :])
                continue

            # Extract video embeddings
            video_start_idx = video_token_indices[0].item()
            video_end_idx = video_token_indices[-1].item() + 1
            num_video_tokens = len(video_token_indices)

            video_embeds = sample_embeds[video_start_idx:video_end_idx]  # [num_video_tokens, hidden_dim]

            # Record original token count
            self.stats['tokens_before_stage1'] = num_video_tokens

            # Get grid info (assume single video per sample)
            t, h, w = video_grid_thw[0].tolist()

            # CRITICAL: Account for Qwen's built-in PatchMerger (2x2 → 1)
            spatial_merge_size = self.model.config.vision_config.spatial_merge_size  # Usually 2
            actual_spatial_per_frame = num_video_tokens // t

            if self.verbose:
                print(f"\n[Stage 1 V2] Sample {i}:")
                print(f"  Grid (t, h, w): ({t}, {h}, {w})")
                print(f"  Expected tokens (pre-PatchMerger): {t * h * w}")
                print(f"  Actual video tokens: {num_video_tokens}")
                print(f"  Spatial tokens per frame: {actual_spatial_per_frame}")
                print(f"  Video token range: [{video_start_idx}, {video_end_idx})")

            # Reshape for processing: [num_tokens, hidden_dim] → [1, t, spatial, hidden_dim]
            video_embeds_reshaped = video_embeds.view(1, t, actual_spatial_per_frame, hidden_dim)

            # Apply Stage 1 merging
            merged_embeds, static_sizes, dynamic_sizes, window_sizes = self.merge_frames_dynamic(
                frames=video_embeds_reshaped,
                num_frames=t,
                num_spatial=actual_spatial_per_frame,
                threshold=self.tau,
                k=7
            )

            # Flatten back: [1, num_merged, hidden_dim] → [num_merged, hidden_dim]
            merged_embeds = merged_embeds.squeeze(0)
            num_merged = merged_embeds.shape[0]

            # Update statistics
            self.stats['tokens_after_stage1'] = num_merged
            self.stats['stage1_compression_ratio'] = 1.0 - (num_merged / num_video_tokens)
            self.stats['num_static_tokens'] = sum(static_sizes)
            self.stats['num_dynamic_tokens'] = sum(dynamic_sizes)

            if self.verbose:
                print(f"  After merging: {num_merged} tokens")
                print(f"  Compression: {self.stats['stage1_compression_ratio']:.1%}")
                print(f"  Static: {self.stats['num_static_tokens']}, Dynamic: {self.stats['num_dynamic_tokens']}")

            # Reconstruct full embeddings: [text_before] + [compressed_video] + [text_after]
            text_before = sample_embeds[:video_start_idx]
            text_after = sample_embeds[video_end_idx:]

            reconstructed = torch.cat([
                text_before,
                merged_embeds,
                text_after
            ], dim=0)

            compressed_embeds_list.append(reconstructed.unsqueeze(0))

            # Record compressed visual token positions (for Stage 2 compatibility)
            # visual_start stays the same (length of text_before)
            # visual_end becomes video_start_idx + num_merged
            compressed_visual_start = video_start_idx
            compressed_visual_end = video_start_idx + num_merged
            self.compressed_visual_positions = (compressed_visual_start, compressed_visual_end)

            if self.verbose:
                print(f"  Compressed visual positions: [{compressed_visual_start}, {compressed_visual_end})")

            # Handle position_ids if present
            if position_ids is not None:
                sample_pos_ids = position_ids[:, i, :]  # [3, seq_len]

                # Keep text position IDs, compress video position IDs
                # For simplicity, we keep the first num_merged position IDs of the video tokens
                # TODO: This is a simplified approach; ideally we'd track which tokens were kept
                pos_before = sample_pos_ids[:, :video_start_idx]
                pos_video = sample_pos_ids[:, video_start_idx:video_start_idx + num_merged]
                pos_after = sample_pos_ids[:, video_end_idx:]

                reconstructed_pos = torch.cat([
                    pos_before,
                    pos_video,
                    pos_after
                ], dim=1)

                compressed_pos_ids_list.append(reconstructed_pos.unsqueeze(1))

        # Pad and concatenate all samples
        # Find max sequence length
        max_new_len = max(emb.shape[1] for emb in compressed_embeds_list)

        # Pad embeddings to same length
        padded_embeds = []
        for emb in compressed_embeds_list:
            if emb.shape[1] < max_new_len:
                padding = torch.zeros(
                    1, max_new_len - emb.shape[1], hidden_dim,
                    device=emb.device, dtype=emb.dtype
                )
                emb = torch.cat([emb, padding], dim=1)
            padded_embeds.append(emb)

        compressed_embeds = torch.cat(padded_embeds, dim=0)

        # Pad position_ids if present
        compressed_position_ids = None
        if position_ids is not None and len(compressed_pos_ids_list) > 0:
            padded_pos_ids = []
            for pos in compressed_pos_ids_list:
                if pos.shape[2] < max_new_len:
                    padding = torch.zeros(
                        3, 1, max_new_len - pos.shape[2],
                        device=pos.device, dtype=pos.dtype
                    )
                    pos = torch.cat([pos, padding], dim=2)
                padded_pos_ids.append(pos)
            compressed_position_ids = torch.cat(padded_pos_ids, dim=1)

        return compressed_embeds, compressed_position_ids

    def _compress_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        new_seq_len: int
    ) -> torch.Tensor:
        """
        Compress attention_mask to match the compressed sequence length.

        Args:
            attention_mask: [batch_size, seq_len] - original attention mask
            input_ids: [batch_size, seq_len] - original input IDs (to locate video tokens)
            new_seq_len: int - new sequence length after compression

        Returns:
            compressed_mask: [batch_size, new_seq_len] - compressed attention mask
        """
        batch_size, seq_len = attention_mask.shape
        video_token_id = self.model.config.video_token_id

        compressed_masks = []

        for i in range(batch_size):
            sample_mask = attention_mask[i]  # [seq_len]
            sample_input_ids = input_ids[i]  # [seq_len]

            # Find video token positions
            video_token_mask = (sample_input_ids == video_token_id)
            video_token_indices = video_token_mask.nonzero(as_tuple=True)[0]

            if len(video_token_indices) == 0:
                # No video tokens, just keep original mask
                compressed_masks.append(sample_mask.unsqueeze(0))
                continue

            # Get video token range
            video_start_idx = video_token_indices[0].item()
            video_end_idx = video_token_indices[-1].item() + 1
            num_video_tokens = len(video_token_indices)

            # Calculate compressed video length
            # new_seq_len = len(text_before) + num_merged + len(text_after)
            # So: num_merged = new_seq_len - len(text_before) - len(text_after)
            len_text_before = video_start_idx
            len_text_after = seq_len - video_end_idx
            num_merged = new_seq_len - len_text_before - len_text_after

            # Reconstruct attention mask
            mask_before = sample_mask[:video_start_idx]
            # For video tokens, keep first num_merged masks (all should be 1)
            mask_video = sample_mask[video_start_idx:video_start_idx + num_merged]
            mask_after = sample_mask[video_end_idx:]

            reconstructed_mask = torch.cat([mask_before, mask_video, mask_after], dim=0)
            compressed_masks.append(reconstructed_mask.unsqueeze(0))

        # Concatenate all masks
        compressed_attention_mask = torch.cat(compressed_masks, dim=0)

        return compressed_attention_mask

    def _compress_cache_position(
        self,
        cache_position: torch.Tensor,
        input_ids: torch.Tensor,
        new_seq_len: int
    ) -> torch.Tensor:
        """
        Compress cache_position to match the compressed sequence length.

        Args:
            cache_position: [seq_len] or [batch_size, seq_len] - original cache positions
            input_ids: [batch_size, seq_len] - original input IDs (to locate video tokens)
            new_seq_len: int - new sequence length after compression

        Returns:
            compressed_position: [new_seq_len] or [batch_size, new_seq_len] - compressed cache positions
        """
        # Check if cache_position is 1D or 2D
        if cache_position.dim() == 1:
            # 1D case: [seq_len] - typically for single batch
            # Simply create new sequential positions for compressed sequence
            return torch.arange(new_seq_len, device=cache_position.device, dtype=cache_position.dtype)

        else:
            # 2D case: [batch_size, seq_len]
            batch_size, seq_len = cache_position.shape
            video_token_id = self.model.config.video_token_id

            compressed_positions = []

            for i in range(batch_size):
                sample_pos = cache_position[i]  # [seq_len]
                sample_input_ids = input_ids[i]  # [seq_len]

                # Find video token positions
                video_token_mask = (sample_input_ids == video_token_id)
                video_token_indices = video_token_mask.nonzero(as_tuple=True)[0]

                if len(video_token_indices) == 0:
                    # No video tokens, create sequential positions
                    compressed_positions.append(
                        torch.arange(new_seq_len, device=cache_position.device, dtype=cache_position.dtype).unsqueeze(0)
                    )
                    continue

                # Get video token range
                video_start_idx = video_token_indices[0].item()
                video_end_idx = video_token_indices[-1].item() + 1

                # Calculate compressed video length
                len_text_before = video_start_idx
                len_text_after = seq_len - video_end_idx
                num_merged = new_seq_len - len_text_before - len_text_after

                # Reconstruct cache position
                pos_before = sample_pos[:video_start_idx]
                # For video tokens, keep first num_merged positions
                pos_video = sample_pos[video_start_idx:video_start_idx + num_merged]
                # For text after, need to adjust positions to be continuous
                pos_after = sample_pos[video_end_idx:]
                # Adjust offset: the gap between before and after needs to account for compression
                if len(pos_after) > 0:
                    offset = (video_start_idx + num_merged) - video_end_idx
                    pos_after = pos_after + offset

                reconstructed_pos = torch.cat([pos_before, pos_video, pos_after], dim=0)
                compressed_positions.append(reconstructed_pos.unsqueeze(0))

            # Concatenate all positions
            compressed_cache_position = torch.cat(compressed_positions, dim=0)
            return compressed_cache_position

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

        (Same implementation as V1 - reused from stage1_wrapper.py)
        """
        B, T, L, C = frames.shape

        # Step 1: Temporal Segmentation
        num_segments = max(1, int(num_frames * self.temporal_segment_ratio))

        frame_features = frames.mean(dim=2)  # [B, T, C]
        idx_clusters, _ = cluster_dpc_knn(frame_features, cluster_num=num_segments, k=k)
        idx_clusters = refine_clusters(idx_clusters)
        window_list = segment_lengths(idx_clusters)

        static_features = []
        dynamic_features = []
        static_sizes = []
        dynamic_sizes = []

        # Step 2: Process each temporal segment
        start_idx = 0
        for window_size in window_list[0]:
            if window_size == 0:
                break

            window_size = window_size.item()
            current_frames = frames[:, start_idx:start_idx+window_size, :, :]

            # Calculate frame similarity
            frames_normed = F.normalize(current_frames, p=2, dim=-1)
            frames_sim = einsum(frames_normed, frames_normed, 'b w l c, b t l c -> b w t l')

            if window_size > 1:
                frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size * (window_size - 1))
            else:
                frames_sim = frames_sim.squeeze(1).squeeze(1)

            # Debug: Print similarity statistics
            if self.verbose:
                print(f"\n[Stage 1 Debug] Frame similarity for window:")
                print(f"  frames_sim shape: {frames_sim.shape}")
                print(f"  frames_sim min: {frames_sim.min().item():.4f}")
                print(f"  frames_sim max: {frames_sim.max().item():.4f}")
                print(f"  frames_sim mean: {frames_sim.mean().item():.4f}")
                print(f"  threshold: {threshold}")

            # Create static/dynamic mask
            mask = frames_sim > threshold

            if self.verbose:
                num_static = mask.sum().item()
                total = mask.numel()
                print(f"  Static tokens: {num_static}/{total} ({num_static/total*100:.1f}%)")
            mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C)

            # Process static regions
            static_mask = mask_expand
            if static_mask.sum() > 0:
                static_feat = torch.masked_select(current_frames, static_mask).view(B, window_size, -1, C).mean(dim=1)

                if static_feat.shape[1] > 14:
                    num_cluster = max(1, int(static_feat.shape[1] * self.cluster_ratio))
                    static_feat = self.spatial_merge_tokens(static_feat, num_cluster=num_cluster, k=7)
            else:
                static_feat = torch.zeros(B, 0, C, device=frames.device, dtype=frames.dtype)

            static_features.append(static_feat)
            static_sizes.append(static_feat.shape[1])

            # Process dynamic regions
            dynamic_mask = ~mask_expand
            if dynamic_mask.sum() > 0:
                dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, window_size, -1, C)

                dynamic_window_list = []
                for i in range(window_size):
                    dynamic_feat_frame = dynamic_feat[:, i, :, :]

                    if dynamic_feat_frame.shape[1] > 14:
                        num_cluster = max(1, int(dynamic_feat_frame.shape[1] * self.cluster_ratio))
                        dynamic_feat_frame = self.spatial_merge_tokens(dynamic_feat_frame, num_cluster=num_cluster, k=7)

                    dynamic_window_list.append(dynamic_feat_frame)

                dynamic_feat = torch.cat(dynamic_window_list, dim=1)
            else:
                dynamic_feat = torch.zeros(B, 0, C, device=frames.device, dtype=frames.dtype)

            dynamic_features.append(dynamic_feat)
            dynamic_sizes.append(dynamic_feat.shape[1])

            start_idx += window_size

        # Concatenate all features
        final_features = []
        for static_feature, dynamic_feature in zip(static_features, dynamic_features):
            final_features.append(static_feature)
            final_features.append(dynamic_feature)
        final_features = torch.cat(final_features, dim=1)

        window_sizes = window_list[0].tolist()
        window_sizes = [w for w in window_sizes if w > 0]

        return final_features, static_sizes, dynamic_sizes, window_sizes

    def spatial_merge_tokens(
        self,
        features: torch.Tensor,
        num_cluster: int,
        k: int = 7
    ) -> torch.Tensor:
        """Apply spatial clustering to merge tokens."""
        cluster_idx, _ = cluster_dpc_knn(features, cluster_num=num_cluster, k=k)
        cluster_features = compute_cluster_vectors(features, cluster_idx, num_cluster=num_cluster)
        return cluster_features

    def set_video_info(self, video_grid_thw: torch.Tensor, pixel_values_videos: torch.Tensor = None):
        """
        Cache video info for use during forward pass.

        Call this before model.generate() or model.forward().
        """
        self.current_video_grid_thw = video_grid_thw
        self.current_pixel_values_videos = pixel_values_videos

    def clear_video_info(self):
        """Clear cached video info after generation."""
        self.current_video_grid_thw = None
        self.current_pixel_values_videos = None
        self.compressed_visual_positions = None  # Clear for next generation

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
        self.compressed_visual_positions = None

    def unhook(self):
        """Restore original model.forward method."""
        if self.original_model_forward is not None:
            self.model.model.forward = self.original_model_forward
            self.original_model_forward = None

            if self.verbose:
                print(f"[Stage 1 Wrapper V2] Unhooked model.forward")
