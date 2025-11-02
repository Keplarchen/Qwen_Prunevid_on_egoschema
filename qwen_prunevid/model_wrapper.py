"""
PruneVid Model Wrapper for Qwen2.5-VL
Provides a unified interface for video inference with token pruning
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

from .prunevid_core import PruneVidCore
from .qwen25_adapter import Qwen25VLAdapter
from .stage1_wrapper import Qwen25VLStage1Wrapper
from .stage1_wrapper_v2 import Qwen25VLStage1WrapperV2


class Qwen25VLPruneVid:
    """
    Qwen2.5-VL model with PruneVid token pruning.

    Usage:
        model = Qwen25VLPruneVid(
            model_path="Qwen2.5-VL-7B-Instruct",
            keep_ratio=0.4,
            enable_pruning=True
        )

        result = model.generate(
            video_path="video.mp4",
            question="What is happening?",
            max_new_tokens=100
        )

        print(f"Answer: {result['generated_text']}")
        print(f"Tokens pruned: {result['pruning_ratio']:.1%}")
    """

    def __init__(
        self,
        model_path: str,
        # Stage 1 parameters
        enable_stage1: bool = False,
        tau: float = 0.8,
        cluster_ratio: float = 0.5,
        temporal_segment_ratio: float = 0.25,
        # Stage 2 parameters
        keep_ratio: float = 0.4,
        pruning_layer: int = 10,
        enable_pruning: bool = True,
        # Model parameters
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        max_frames: int = 16,
        min_frames: int = 4,
        fps: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Args:
            model_path: Path to Qwen2.5-VL model
            enable_stage1: Enable Stage 1 Spatial-Temporal Token Merging
            tau: Static/dynamic threshold for Stage 1 (0.7-0.9, default: 0.8)
            cluster_ratio: Spatial clustering ratio for Stage 1 (0.3-0.8, default: 0.5)
            temporal_segment_ratio: Temporal segmentation ratio for Stage 1 (0.25-0.5, default: 0.25)
            keep_ratio: Ratio of visual tokens to keep in Stage 2 (default: 0.4 = 40%)
            pruning_layer: Which LLM layer to prune at in Stage 2 (default: 10)
            enable_pruning: Enable Stage 2 Attention-Based Pruning
            device: Device to run on
            torch_dtype: Model dtype
            max_frames: Maximum number of frames to sample
            min_frames: Minimum number of frames to sample
            fps: Frame sampling rate (None = uniform sampling)
            verbose: Print debug information
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_stage1 = enable_stage1
        self.enable_pruning = enable_pruning
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.fps = fps
        self.verbose = verbose

        # Load model and processor
        print(f"Loading Qwen2.5-VL model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else device,
            attn_implementation="eager"  # Required to access attention weights
        )

        self.model.eval()

        # Initialize Stage 1 (Spatial-Temporal Token Merging)
        # V2: Token merging happens in LLM model.forward (after validation)
        if self.enable_stage1:
            self.stage1_wrapper = Qwen25VLStage1WrapperV2(
                model=self.model,
                tau=tau,
                cluster_ratio=cluster_ratio,
                temporal_segment_ratio=temporal_segment_ratio,
                enable_stage1=True,
                verbose=verbose
            )
            print(f"✓ Stage 1 V2 enabled (tau={tau}, cluster_ratio={cluster_ratio}, temporal_segment_ratio={temporal_segment_ratio})")
            print(f"  → Token merging in LLM forward (TimeChat-Online style)")
        else:
            self.stage1_wrapper = None
            if verbose:
                print("✓ Stage 1 disabled")

        # Initialize PruneVid components
        if self.enable_pruning:
            self.prunevid_core = PruneVidCore(
                keep_ratio=keep_ratio,
                pruning_layer=pruning_layer,
                enable_stage1=False,  # Qwen2.5-VL has built-in PatchMerger
                verbose=verbose
            )

            self.adapter = Qwen25VLAdapter(
                model=self.model,
                prunevid_core=self.prunevid_core,
                stage1_wrapper=self.stage1_wrapper,  # Pass Stage 1 wrapper for token position detection
                verbose=verbose
            )

            # Register hooks
            self.adapter.register_pruning_hook()

            print(f"✓ Stage 2 enabled (keep_ratio={keep_ratio}, layer={pruning_layer})")
        else:
            self.prunevid_core = None
            self.adapter = None
            if not self.enable_stage1:
                print("✓ Running in baseline mode (no pruning)")

    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load and sample video frames.

        Args:
            video_path: Path to video file

        Returns:
            frames: [num_frames, H, W, C] numpy array
        """
        if not DECORD_AVAILABLE:
            raise RuntimeError("decord is required. Install: pip install decord")

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        video_duration = total_frames / video_fps

        # Determine number of frames to sample
        if self.fps is not None:
            # Sample at specific FPS
            target_num_frames = int(video_duration * self.fps)
        else:
            # Uniform sampling up to max_frames
            target_num_frames = min(total_frames, self.max_frames)

        # Clamp to [min_frames, max_frames]
        target_num_frames = max(self.min_frames, min(target_num_frames, self.max_frames))

        # Sample frame indices
        if total_frames <= target_num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, target_num_frames, dtype=int)

        frames = vr.get_batch(indices).asnumpy()

        if self.verbose:
            print(f"\n[Video Loading]")
            print(f"  Path: {video_path}")
            print(f"  Duration: {video_duration:.1f}s, FPS: {video_fps:.1f}")
            print(f"  Sampled: {len(indices)} frames")

        return frames

    def generate(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer for a video question.

        Args:
            video_path: Path to video file
            question: Question text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Dictionary with:
                - generated_text: Generated answer
                - num_frames: Number of video frames
                - tokens_before: Sequence length before pruning
                - tokens_after: Sequence length after pruning
                - pruning_ratio: Ratio of tokens pruned
                - inference_time: Time taken for inference
        """
        import time

        start_time = time.time()

        # Load video
        video_frames = self.load_video(video_path)
        num_frames = len(video_frames)
        load_time = time.time() - start_time

        # Prepare messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": question}
            ]
        }]

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            videos=[video_frames],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if self.verbose:
            print(f"\n[Input Processing]")
            print(f"  Input IDs shape: {inputs['input_ids'].shape}")
            if 'image_grid_thw' in inputs:
                print(f"  image_grid_thw: {inputs['image_grid_thw']}")
            if 'video_grid_thw' in inputs:
                print(f"  video_grid_thw: {inputs['video_grid_thw']}")

        # Set video info for Stage 1 V2 (needed for LLM forward hook)
        if self.enable_stage1 and 'video_grid_thw' in inputs:
            pixel_values_videos = inputs.get('pixel_values_videos', None)
            self.stage1_wrapper.set_video_info(
                video_grid_thw=inputs['video_grid_thw'],
                pixel_values_videos=pixel_values_videos
            )
            if self.verbose:
                print(f"\n[Model Wrapper] Set video info for Stage 1:")
                print(f"  video_grid_thw: {inputs['video_grid_thw']}")
        elif self.enable_stage1:
            if self.verbose:
                print(f"\n[Model Wrapper] Warning: video_grid_thw not found in inputs!")
                print(f"  Available keys: {list(inputs.keys())}")

        # Prepare for generation with pruning
        past_key_values = None
        if self.enable_pruning:
            # 创建自定义cache
            past_key_values = self.adapter.create_custom_cache()
            self.adapter.prepare_for_generation(inputs, num_frames)

        # Generate
        inference_start = time.time()

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,  # 使用自定义cache
                output_attentions=True,  # 必须启用！否则无法获取attention权重
                do_sample=False,
                **kwargs
            )

        inference_time = time.time() - inference_start

        # Clear video info for Stage 1 V2
        if self.enable_stage1:
            self.stage1_wrapper.clear_video_info()

        # Finish generation
        if self.enable_pruning:
            self.adapter.finish_generation()

        # Decode output
        generated_text = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Collect statistics
        stats = {}

        # Stage 1 statistics
        if self.enable_stage1:
            stage1_stats = self.stage1_wrapper.get_stats()
            stats.update({
                'tokens_before_stage1': stage1_stats.get('tokens_before_stage1', 0),
                'tokens_after_stage1': stage1_stats.get('tokens_after_stage1', 0),
                'stage1_compression_ratio': stage1_stats.get('stage1_compression_ratio', 0.0),
                'num_static_tokens': stage1_stats.get('num_static_tokens', 0),
                'num_dynamic_tokens': stage1_stats.get('num_dynamic_tokens', 0),
            })

        # Stage 2 statistics
        if self.enable_pruning:
            stage2_stats = self.adapter.get_stats()
            stats.update(stage2_stats)

            # Calculate total compression ratio (Stage 1 + Stage 2)
            if self.enable_stage1 and stats.get('tokens_before_stage1', 0) > 0:
                original_tokens = stats['tokens_before_stage1']
                final_tokens = stats.get('tokens_after', stats.get('tokens_after_stage1', original_tokens))
                stats['total_compression_ratio'] = 1.0 - (final_tokens / original_tokens)
            elif stats.get('tokens_before', 0) > 0:
                stats['total_compression_ratio'] = stats.get('pruning_ratio', 0.0)

        result = {
            'generated_text': generated_text,
            'num_frames': num_frames,
            'input_tokens': inputs['input_ids'].shape[1],
            'output_tokens': output.shape[1] - inputs['input_ids'].shape[1],
            'video_load_time': load_time,
            'inference_time': inference_time,
            'total_time': time.time() - start_time,
            **stats
        }

        if self.verbose:
            print(f"\n[Generation Result]")
            print(f"  Generated: {generated_text}")
            print(f"  Time: {inference_time:.2f}s")

            # Display Stage 1 statistics
            if self.enable_stage1 and stats.get('tokens_before_stage1', 0) > 0:
                print(f"  Stage 1: {stats['tokens_before_stage1']} → {stats['tokens_after_stage1']} "
                      f"({stats['stage1_compression_ratio']:.1%})")
                print(f"    Static tokens: {stats['num_static_tokens']}, "
                      f"Dynamic tokens: {stats['num_dynamic_tokens']}")

            # Display Stage 2 statistics
            if self.enable_pruning and stats.get('pruning_applied', False):
                print(f"  Stage 2: {stats['tokens_before']} → {stats['tokens_after']} "
                      f"({stats['pruning_ratio']:.1%})")
                print(f"    Visual: {stats['num_visual_tokens']} → {stats['num_visual_kept']} "
                      f"({stats['visual_pruning_ratio']:.1%})")

            # Display total compression
            if stats.get('total_compression_ratio', 0) > 0:
                print(f"  Total compression: {stats['total_compression_ratio']:.1%}")

        return result

    def process_egoschema_sample(
        self,
        video_path: str,
        question: str,
        options: List[str]
    ) -> tuple:
        """
        Process an EgoSchema multiple-choice sample.

        Args:
            video_path: Path to video file
            question: Question text
            options: List of 5 option strings (A-E)

        Returns:
            (prediction, generated_text, stats_dict)
        """
        # Format prompt with options
        prompt = f"{question}\n\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}. {opt}\n"
        prompt += "\nAnswer with the option's letter from the given choices directly."

        # Generate
        result = self.generate(video_path, prompt, max_new_tokens=10)

        # Extract answer letter
        generated_text = result['generated_text']
        prediction = None
        for char in generated_text.upper():
            if char in 'ABCDE':
                prediction = char
                break

        return prediction, generated_text, result

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        if self.adapter is not None:
            return self.adapter.get_stats()
        return {}

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'adapter') and self.adapter is not None:
            self.adapter.remove_hooks()
        if hasattr(self, 'stage1_wrapper') and self.stage1_wrapper is not None:
            self.stage1_wrapper.unhook()
