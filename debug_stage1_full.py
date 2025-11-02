#!/usr/bin/env python3
"""
Full debug script to trace Stage 1 execution
"""

import os
import sys
from pathlib import Path

# Set GPU before imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from qwen_prunevid import Qwen25VLPruneVid

# Configuration
MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "/mnt/ssd_ext/huggingface/egoschema/videos/2f814211-32de-4e8f-b26e-096ab47d20e8.mp4"

print("=" * 80)
print("Full Debug: Stage 1 Execution Trace")
print("=" * 80)

# Initialize model with Stage 1 enabled and verbose output
print("\nInitializing model with Stage 1 enabled (verbose=True)...")
model = Qwen25VLPruneVid(
    model_path=MODEL_PATH,
    # Stage 1 parameters
    enable_stage1=True,
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
    # Stage 2 parameters
    enable_pruning=False,  # Disable Stage 2 for now to isolate Stage 1
    # Other parameters
    device="cuda",
    max_frames=16,
    verbose=True  # Enable verbose output
)

print("\n" + "=" * 80)
print("Running generation...")
print("=" * 80)

# Generate
result = model.generate(
    video_path=VIDEO_PATH,
    question="What is happening in this video?",
    max_new_tokens=50
)

print("\n" + "=" * 80)
print("Results:")
print("=" * 80)

print(f"\nGenerated text: {result['generated_text']}")
print(f"\nNum frames: {result['num_frames']}")

# Check Stage 1 statistics
print("\n" + "-" * 80)
print("Stage 1 Statistics:")
print("-" * 80)

if 'tokens_before_stage1' in result:
    print(f"  Tokens before Stage 1: {result['tokens_before_stage1']}")
    print(f"  Tokens after Stage 1:  {result['tokens_after_stage1']}")
    print(f"  Stage 1 compression:   {result['stage1_compression_ratio']:.2%}")
    print(f"  Static tokens:         {result.get('num_static_tokens', 0)}")
    print(f"  Dynamic tokens:        {result.get('num_dynamic_tokens', 0)}")
else:
    print("  ❌ NO Stage 1 statistics found in result!")
    print(f"\n  Available keys in result:")
    for key in sorted(result.keys()):
        print(f"    - {key}: {result[key]}")

# Get stats directly from wrapper
print("\n" + "-" * 80)
print("Stage 1 Wrapper Stats (direct query):")
print("-" * 80)

if model.stage1_wrapper:
    wrapper_stats = model.stage1_wrapper.get_stats()
    print(f"  Tokens before: {wrapper_stats.get('tokens_before_stage1', 0)}")
    print(f"  Tokens after:  {wrapper_stats.get('tokens_after_stage1', 0)}")
    print(f"  Compression:   {wrapper_stats.get('stage1_compression_ratio', 0.0):.2%}")
    print(f"  Static:        {wrapper_stats.get('num_static_tokens', 0)}")
    print(f"  Dynamic:       {wrapper_stats.get('num_dynamic_tokens', 0)}")
else:
    print("  ❌ stage1_wrapper is None!")

print("\n" + "=" * 80)
print("Diagnosis:")
print("=" * 80)

if result.get('tokens_before_stage1', 0) > 0:
    print("✓ Stage 1 was applied successfully")
elif result.get('stage1_compression_ratio', 0.0) > 0:
    print("✓ Stage 1 was applied (compression detected)")
else:
    print("❌ Stage 1 was NOT applied")
    print("\nPossible reasons:")
    print("  1. apply_stage1 condition was False in wrapped_model_forward")
    print("  2. Hook was not triggered")
    print("  3. Statistics were not properly collected")
    print("\nCheck the verbose output above for clues.")
