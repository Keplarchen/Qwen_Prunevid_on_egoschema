"""
PruneVid Core Implementation
Based on: "PruneVid: Visual Token Pruning for Efficient Video Large Language Models"
Paper: https://arxiv.org/abs/2412.16117
Official Code: https://github.com/Visual-AI/PruneVid
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class PruneVidCore:
    """
    Core PruneVid implementation following the paper's three-stage approach:

    Stage 1: Spatial-Temporal Token Merging (optional, vision encoder level)
    Stage 2: Attention-Based Token Pruning (LLM level)
    Stage 3: KV Cache Compression (all subsequent layers)
    """

    def __init__(
        self,
        keep_ratio: float = 0.4,
        pruning_layer: int = 10,
        enable_stage1: bool = False,  # Usually not needed for Qwen2.5-VL
        verbose: bool = False
    ):
        """
        Args:
            keep_ratio: Ratio of visual tokens to keep (paper default: 0.4 = 40%)
            pruning_layer: Which LLM layer to perform pruning at (paper default: 10)
            enable_stage1: Enable vision encoder merging (optional for Qwen2.5-VL)
            verbose: Print debug information
        """
        self.keep_ratio = keep_ratio
        self.pruning_layer = pruning_layer
        self.enable_stage1 = enable_stage1
        self.verbose = verbose

        # Statistics
        self.stats = {
            'tokens_before': 0,
            'tokens_after': 0,
            'num_visual_tokens': 0,
            'num_kept_visual_tokens': 0,
            'pruning_ratio': 0.0
        }

    def reset_stats(self):
        """Reset statistics for new generation."""
        self.stats = {
            'tokens_before': 0,
            'tokens_after': 0,
            'num_visual_tokens': 0,
            'num_kept_visual_tokens': 0,
            'pruning_ratio': 0.0
        }

    def compute_token_importance_attention(
        self,
        attention_weights: torch.Tensor,
        visual_start: int,
        visual_end: int,
        text_start: int
    ) -> torch.Tensor:
        """
        Compute visual token importance using text-to-visual attention.

        This is the core of PruneVid Stage 2.

        Args:
            attention_weights: [B, num_heads, seq_len, seq_len]
            visual_start: Start index of visual tokens
            visual_end: End index of visual tokens
            text_start: Start index of text tokens

        Returns:
            importance: [B, num_visual_tokens] importance scores
        """
        # Extract text-to-visual attention
        # text_to_visual: [B, num_heads, num_text_tokens, num_visual_tokens]
        text_to_visual = attention_weights[:, :, text_start:, visual_start:visual_end]

        if self.verbose:
            print(f"    [PruneVid] Text-to-visual attention shape: {text_to_visual.shape}")

        # Aggregate: max over text dimension (which visual token gets most attention from ANY text token)
        # Then average over heads
        # [B, num_heads, num_visual_tokens]
        max_attention_per_visual = text_to_visual.max(dim=2)[0]

        # [B, num_visual_tokens]
        importance = max_attention_per_visual.mean(dim=1)

        if self.verbose:
            print(f"    [PruneVid] Importance scores - min: {importance.min():.4f}, "
                  f"max: {importance.max():.4f}, mean: {importance.mean():.4f}")

        return importance

    def compute_token_importance_norm(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int
    ) -> torch.Tensor:
        """
        Fallback: Compute importance based on feature norm.

        Args:
            hidden_states: [B, seq_len, hidden_dim]
            visual_start: Start index of visual tokens
            visual_end: End index of visual tokens

        Returns:
            importance: [B, num_visual_tokens]
        """
        visual_hidden = hidden_states[:, visual_start:visual_end, :]
        importance = visual_hidden.norm(dim=-1)  # [B, num_visual_tokens]

        if self.verbose:
            print(f"    [PruneVid] Using norm-based importance (fallback)")

        return importance

    def select_important_tokens(
        self,
        importance: torch.Tensor,
        num_visual: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Select top-k important visual tokens.

        Args:
            importance: [B, num_visual_tokens]
            num_visual: Total number of visual tokens

        Returns:
            kept_indices: [B, num_keep] indices of kept tokens (sorted)
            num_keep: Number of tokens to keep
        """
        num_keep = max(1, int(num_visual * self.keep_ratio))

        if self.verbose:
            print(f"    [PruneVid] Selecting top {num_keep}/{num_visual} tokens "
                  f"({self.keep_ratio:.1%})")

        # Get top-k indices
        # topk_indices: [B, num_keep]
        _, topk_indices = torch.topk(importance, k=num_keep, dim=-1, largest=True)

        # Sort indices to maintain spatial/temporal order
        topk_indices = topk_indices.sort(dim=-1)[0]

        return topk_indices, num_keep

    def reconstruct_sequence(
        self,
        hidden_states: torch.Tensor,
        kept_indices: torch.Tensor,
        visual_start: int,
        visual_end: int,
        text_start: int
    ) -> torch.Tensor:
        """
        Reconstruct sequence with kept visual tokens + all text tokens.

        Args:
            hidden_states: [B, seq_len, hidden_dim]
            kept_indices: [B, num_keep]
            visual_start: Start index of visual tokens
            visual_end: End index of visual tokens
            text_start: Start index of text tokens

        Returns:
            new_hidden: [B, num_keep + num_text, hidden_dim]
        """
        B, seq_len, hidden_dim = hidden_states.shape
        num_keep = kept_indices.shape[1]

        # Adjust indices to absolute positions
        kept_indices_abs = kept_indices + visual_start  # [B, num_keep]

        # Gather kept visual tokens
        # Create batch indices: [B, num_keep]
        batch_indices = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, num_keep)

        # Gather: [B, num_keep, hidden_dim]
        kept_visual = hidden_states[batch_indices, kept_indices_abs, :]

        # Get all text tokens
        text_hidden = hidden_states[:, text_start:, :]  # [B, num_text, hidden_dim]

        # Concatenate: [kept_visual_tokens, text_tokens]
        new_hidden = torch.cat([kept_visual, text_hidden], dim=1)

        if self.verbose:
            print(f"    [PruneVid] Sequence reconstructed: {seq_len} → {new_hidden.shape[1]} tokens")
            print(f"    [PruneVid] Visual: {visual_end - visual_start} → {num_keep}, "
                  f"Text: {seq_len - text_start}")

        return new_hidden

    def prune_tokens(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main pruning function - Stage 2 of PruneVid.

        Args:
            hidden_states: [B, seq_len, hidden_dim]
            visual_start: Start index of visual tokens
            visual_end: End index of visual tokens
            attention_weights: Optional [B, num_heads, seq_len, seq_len]

        Returns:
            new_hidden: [B, new_seq_len, hidden_dim] pruned sequence
            stats: Dictionary with pruning statistics
        """
        B, seq_len, hidden_dim = hidden_states.shape
        num_visual = visual_end - visual_start
        text_start = visual_end

        if num_visual == 0:
            if self.verbose:
                print(f"    [PruneVid] No visual tokens found, skipping pruning")
            return hidden_states, {}

        # Compute importance scores
        if attention_weights is not None:
            importance = self.compute_token_importance_attention(
                attention_weights, visual_start, visual_end, text_start
            )
        else:
            importance = self.compute_token_importance_norm(
                hidden_states, visual_start, visual_end
            )

        # Select important tokens
        kept_indices, num_keep = self.select_important_tokens(importance, num_visual)

        # Reconstruct sequence
        new_hidden = self.reconstruct_sequence(
            hidden_states, kept_indices, visual_start, visual_end, text_start
        )

        # Update statistics
        self.stats['tokens_before'] = seq_len
        self.stats['tokens_after'] = new_hidden.shape[1]
        self.stats['num_visual_tokens'] = num_visual
        self.stats['num_kept_visual_tokens'] = num_keep
        self.stats['pruning_ratio'] = 1.0 - (new_hidden.shape[1] / seq_len)

        return new_hidden, {
            'kept_indices': kept_indices,
            'num_visual_before': num_visual,
            'num_visual_after': num_keep,
            'new_visual_end': num_keep,  # Visual tokens now at [0:num_keep]
            **self.stats
        }

    # compress_kv_cache方法已被移除
    # 现在由PruneVidDynamicCache类在其update()方法中自动处理KV cache压缩
    # 见 prunevid_cache.py 的实现

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.stats.copy()
