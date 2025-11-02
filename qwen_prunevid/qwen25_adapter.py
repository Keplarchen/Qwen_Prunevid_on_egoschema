"""
Qwen2.5-VL Adapter for PruneVid
Handles Qwen2.5-VL specific architecture details
"""

import torch
from typing import Tuple, Optional, Any
from .prunevid_core import PruneVidCore
from .prunevid_cache import PruneVidDynamicCache


class Qwen25VLAdapter:
    """
    Adapter for integrating PruneVid with Qwen2.5-VL model.

    Handles:
    - Visual token position detection using image_grid_thw
    - Hook registration at LLM layers
    - Attention weight extraction from Qwen2.5-VL's attention module
    - Architecture-specific modifications (RoPE, GQA, etc.)
    """

    def __init__(
        self,
        model: Any,
        prunevid_core: PruneVidCore,
        stage1_wrapper: Any = None,
        verbose: bool = False
    ):
        """
        Args:
            model: Qwen2_5_VLForConditionalGeneration instance
            prunevid_core: PruneVidCore instance
            stage1_wrapper: Optional Stage1 wrapper (for getting actual visual token count)
            verbose: Print debug information
        """
        self.model = model
        self.prunevid = prunevid_core
        self.stage1_wrapper = stage1_wrapper
        self.verbose = verbose

        # State
        self.visual_token_positions = None
        self.cached_attention = None
        self.kept_visual_indices = None  # Indices of kept visual tokens
        self.is_pruning_active = False
        self.hook_handles = []
        self.custom_cache = None  # PruneVidDynamicCache instance

        # Verify model structure
        self._verify_model_structure()

    def _verify_model_structure(self):
        """Verify that the model has expected Qwen2.5-VL structure."""
        assert hasattr(self.model, 'model'), "Model should have 'model' attribute"
        assert hasattr(self.model.model, 'language_model'), \
            "Model should have language_model attribute"
        assert hasattr(self.model.model.language_model, 'layers'), \
            "Language model should have layers"

        num_layers = len(self.model.model.language_model.layers)

        if self.verbose:
            print(f"\n[Qwen2.5-VL Adapter] Model verified:")
            print(f"  Type: {type(self.model).__name__}")
            print(f"  LLM layers: {num_layers}")
            print(f"  Pruning layer: {self.prunevid.pruning_layer}")

    def detect_visual_token_positions(
        self,
        inputs: dict,
        num_frames: int
    ) -> Tuple[int, int]:
        """
        Detect visual token positions using image_grid_thw.

        This is the CORRECT way for Qwen2.5-VL, which provides exact token counts.

        IMPORTANT: If Stage 1 is active, use its actual output count instead of
        calculating from image_grid_thw (which reflects pre-Stage 1 count).

        Args:
            inputs: Processor outputs containing image_grid_thw
            num_frames: Number of video frames

        Returns:
            (visual_start, visual_end): Token position range
        """
        # Method 1: Use image_grid_thw (ACCURATE)
        image_grid_thw = inputs.get('image_grid_thw', None)

        if image_grid_thw is not None:
            # image_grid_thw: [num_images, 3] where 3 = [temporal, height, width]
            # After Qwen2.5-VL's PatchMerger (2×2 spatial pooling):
            # num_tokens = temporal × (height/2) × (width/2)

            total_visual_tokens = 0
            for i in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[i].tolist()
                # PatchMerger does 2×2 spatial pooling
                tokens = int(t * (h // 2) * (w // 2))
                total_visual_tokens += tokens

            if self.verbose:
                print(f"\n[Token Detection] Using image_grid_thw:")
                print(f"  image_grid_thw shape: {image_grid_thw.shape}")
                print(f"  image_grid_thw values: {image_grid_thw}")
                print(f"  Total visual tokens (pre-Stage 1): {total_visual_tokens}")

            # CRITICAL FIX: If Stage 1 is active, use its actual output count
            # Stage 1 compresses tokens, so the actual count is different from image_grid_thw
            if self.stage1_wrapper is not None and self.stage1_wrapper.enable_stage1:
                stage1_stats = self.stage1_wrapper.get_stats()
                actual_tokens = stage1_stats.get('tokens_after_stage1', 0)

                if actual_tokens > 0:
                    if self.verbose:
                        print(f"  ✅ Stage 1 active: {total_visual_tokens} → {actual_tokens} tokens")
                    total_visual_tokens = actual_tokens
                else:
                    if self.verbose:
                        print(f"  ⚠️  Stage 1 enabled but no stats yet, using pre-Stage 1 count")

            return (0, total_visual_tokens)

        # Method 2: Fallback estimation (LESS ACCURATE)
        else:
            if self.verbose:
                print(f"\n[Token Detection] image_grid_thw not found, using estimation")

            # Based on testing: ~111 tokens per frame for Qwen2.5-VL
            # This varies with video resolution
            estimated_tokens_per_frame = 111
            estimated_total = num_frames * estimated_tokens_per_frame

            # Limit to reasonable range (assuming some text tokens exist)
            seq_len = inputs['input_ids'].shape[1]
            estimated_total = min(estimated_total, seq_len - 10)

            if self.verbose:
                print(f"  Estimated: {num_frames} frames × {estimated_tokens_per_frame} = {estimated_total}")
                print(f"  ⚠️  This is an estimate and may be inaccurate!")

            return (0, estimated_total)

    def register_pruning_hook(self):
        """
        注册forward hook以计算要保留的visual tokens

        核心逻辑：
        1. 在pruning_layer完成时，根据attention权重计算kept_visual_indices
        2. 将indices配置到custom_cache中
        3. Cache会自动在update时应用pruning（真正删除token）

        不再需要在后续层添加hook，因为cache已经被压缩了
        """
        llm = self.model.model.language_model
        num_layers = len(llm.layers)
        target_layer_idx = self.prunevid.pruning_layer - 1  # Convert to 0-indexed

        if self.verbose:
            print(f"\n[Hook Registration] PruneVid hook:")
            print(f"  Total LLM layers: {num_layers}")
            print(f"  Pruning at layer: {self.prunevid.pruning_layer} (0-indexed: {target_layer_idx})")
            print(f"  Strategy: Custom cache will auto-compress at layer {target_layer_idx}")

        # 重置状态
        self.kept_visual_indices = None

        def compute_kept_indices_hook(module, input, output):
            """
            在pruning_layer计算哪些visual token要保留
            并配置custom_cache的pruning参数
            """
            if not self.is_pruning_active:
                return output

            # 只计算一次
            if self.kept_visual_indices is not None:
                return output

            if self.verbose:
                print(f"\n[Layer {self.prunevid.pruning_layer}] Computing token importance...")

            # 提取hidden states和attention weights
            if isinstance(output, tuple) and len(output) > 1:
                hidden_states = output[0]
                attention_weights = output[1]
            else:
                hidden_states = output if not isinstance(output, tuple) else output[0]
                attention_weights = None

            # 检查visual token位置
            # CRITICAL: If Stage 1 V2 is active, use its compressed positions
            if self.stage1_wrapper is not None and hasattr(self.stage1_wrapper, 'compressed_visual_positions'):
                compressed_pos = self.stage1_wrapper.compressed_visual_positions
                if compressed_pos is not None:
                    visual_start, visual_end = compressed_pos
                    if self.verbose:
                        print(f"  ✅ Using Stage 1 V2 compressed positions: [{visual_start}, {visual_end})")
                elif self.visual_token_positions is not None:
                    visual_start, visual_end = self.visual_token_positions
                    if self.verbose:
                        print(f"  ⚠️  Stage 1 V2 positions not ready, using pre-computed: [{visual_start}, {visual_end})")
                else:
                    if self.verbose:
                        print(f"  ⚠️  No visual tokens, skipping pruning")
                    return output
            elif self.visual_token_positions is None:
                if self.verbose:
                    print(f"  ⚠️  No visual tokens, skipping pruning")
                return output
            else:
                visual_start, visual_end = self.visual_token_positions

            num_visual = visual_end - visual_start

            if num_visual == 0:
                if self.verbose:
                    print(f"  ⚠️  Zero visual tokens, skipping")
                return output

            # 计算token importance
            if attention_weights is not None:
                text_start = visual_end
                # text-to-visual attention: [B, num_heads, num_text, num_visual]
                text_to_visual = attention_weights[:, :, text_start:, visual_start:visual_end]

                # 重要：按照PLLaVA和论文，应该是 max over text, max over heads
                # 而不是 max over text, mean over heads
                # Result: [B, num_heads, num_visual] -> [B, num_visual] -> [num_visual]
                importance = text_to_visual.max(dim=2)[0].max(dim=1)[0]

                # Remove batch dimension (assume batch_size=1)
                if importance.dim() > 1:
                    importance = importance[0]  # [num_visual]

                if self.verbose:
                    print(f"  Using attention-based importance")
                    print(f"  Attention shape: {attention_weights.shape}")
                    print(f"  Text-to-visual shape: {text_to_visual.shape}")
                    print(f"  Importance shape: {importance.shape}")
            else:
                # Fallback: 使用hidden states的norm
                visual_hidden = hidden_states[:, visual_start:visual_end, :]
                importance = visual_hidden.norm(dim=-1)

                # Remove batch dimension (assume batch_size=1)
                if importance.dim() > 1:
                    importance = importance[0]  # [num_visual]

                if self.verbose:
                    print(f"  ⚠️  No attention weights, using norm-based fallback")

            # Top-k selection
            num_keep = max(1, int(num_visual * self.prunevid.keep_ratio))
            _, topk_indices = torch.topk(importance, k=num_keep, largest=True)
            topk_indices = topk_indices.sort()[0]  # 排序保持时空顺序

            # 保存indices
            self.kept_visual_indices = topk_indices

            # 配置custom_cache的pruning
            if self.custom_cache is not None:
                self.custom_cache.configure_pruning(
                    pruning_layer=target_layer_idx,
                    kept_visual_indices=topk_indices,
                    visual_start=visual_start,
                    visual_end=visual_end
                )

                if self.verbose:
                    print(f"  ✅ Configured cache pruning at layer {target_layer_idx}")
            else:
                if self.verbose:
                    print(f"  ⚠️  No custom_cache, pruning will not be applied!")

            # 更新统计信息
            self.prunevid.stats['tokens_before'] = hidden_states.shape[1]
            self.prunevid.stats['tokens_after'] = num_keep + (hidden_states.shape[1] - num_visual)
            self.prunevid.stats['num_visual_tokens'] = num_visual
            self.prunevid.stats['num_kept_visual_tokens'] = num_keep
            self.prunevid.stats['pruning_ratio'] = 1.0 - (self.prunevid.stats['tokens_after'] / self.prunevid.stats['tokens_before'])

            if self.verbose:
                print(f"  Visual tokens: {num_visual} → {num_keep} ({self.prunevid.keep_ratio:.1%})")
                print(f"  Total tokens: {self.prunevid.stats['tokens_before']} → {self.prunevid.stats['tokens_after']}")
                print(f"  Pruning ratio: {self.prunevid.stats['pruning_ratio']:.1%}")

            return output

        # 只注册一个hook到pruning_layer
        handle = llm.layers[target_layer_idx].register_forward_hook(compute_kept_indices_hook)
        self.hook_handles.append(handle)

        if self.verbose:
            print(f"  ✅ Registered 1 hook at layer {target_layer_idx}")

    def create_custom_cache(self):
        """
        创建PruneVidDynamicCache实例

        Returns:
            PruneVidDynamicCache: 支持自动pruning的cache
        """
        self.custom_cache = PruneVidDynamicCache()

        if self.verbose:
            print(f"\n[Cache] Created PruneVidDynamicCache")

        return self.custom_cache

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        if self.verbose:
            print(f"\n[Hooks] Removed all hooks")

    def prepare_for_generation(
        self,
        inputs: dict,
        num_frames: int
    ):
        """
        Prepare for generation with pruning.

        Args:
            inputs: Processor outputs
            num_frames: Number of video frames
        """
        # Detect visual token positions
        self.visual_token_positions = self.detect_visual_token_positions(inputs, num_frames)

        # Reset pruning state
        self.prunevid.reset_stats()
        self.kept_visual_indices = None  # Reset for new generation
        self.cached_attention = None

        # Reset custom cache if exists
        if self.custom_cache is not None:
            self.custom_cache.reset_pruning_state()

        # Activate pruning
        self.is_pruning_active = True

        if self.verbose:
            visual_start, visual_end = self.visual_token_positions
            print(f"\n[Generation Prep] Ready for pruning:")
            print(f"  Visual tokens: [{visual_start}:{visual_end}] ({visual_end - visual_start} tokens)")
            print(f"  Pruning active: {self.is_pruning_active}")

    def finish_generation(self):
        """Clean up after generation."""
        self.is_pruning_active = False
        self.cached_attention = None
        self.kept_visual_indices = None

        if self.verbose:
            # 优先从custom_cache获取统计信息（更准确）
            if self.custom_cache is not None:
                cache_stats = self.custom_cache.get_pruning_stats()
                if cache_stats.get('pruning_applied', False):
                    print(f"\n[Generation Complete] Pruning stats from cache:")
                    print(f"  Tokens: {cache_stats['tokens_before']} → {cache_stats['tokens_after']}")
                    print(f"  Overall pruning ratio: {cache_stats['pruning_ratio']:.1%}")
                    print(f"  Visual tokens: {cache_stats['num_visual_tokens']} → {cache_stats['num_visual_kept']}")
                    print(f"  Visual pruning ratio: {cache_stats['visual_pruning_ratio']:.1%}")
                else:
                    print(f"\n[Generation Complete] No pruning occurred (cache)")
            else:
                # Fallback to prunevid core stats
                stats = self.prunevid.get_stats()
                if stats.get('tokens_before', 0) > 0:
                    print(f"\n[Generation Complete]")
                    print(f"  Tokens: {stats['tokens_before']} → {stats['tokens_after']}")
                    print(f"  Pruning ratio: {stats['pruning_ratio']:.1%}")
                else:
                    print(f"\n[Generation Complete] No pruning occurred")

    def get_stats(self) -> dict:
        """
        Get pruning statistics.

        优先返回custom_cache的统计（更准确），否则返回prunevid core的统计
        """
        if self.custom_cache is not None:
            return self.custom_cache.get_pruning_stats()
        else:
            return self.prunevid.get_stats()
