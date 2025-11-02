#!/usr/bin/env python3
"""
Test script for Stage 1 V2 implementation
Tests that the new implementation resolves token validation issues
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from qwen_prunevid import Qwen25VLPruneVid


def test_stage1_v2():
    """Test Stage 1 V2 with a simple video."""
    print("="*80)
    print("Testing Stage 1 V2 Implementation")
    print("="*80)

    # Configuration
    MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"
    VIDEO_PATH = "egoschema/videos/0050717e-e13e-49bc-a5aa-d3fb66d1c90f.mp4"  # Use a test video

    print(f"\n📦 Model: {MODEL_PATH}")
    print(f"🎬 Video: {VIDEO_PATH}")

    # Initialize model with Stage 1 V2
    print(f"\n🚀 Initializing model with Stage 1 V2...")
    try:
        model = Qwen25VLPruneVid(
            model_path=MODEL_PATH,
            # Stage 1 V2 parameters
            enable_stage1=True,
            tau=0.8,
            cluster_ratio=0.5,
            temporal_segment_ratio=0.25,
            # Disable Stage 2 for this test
            enable_pruning=False,
            # Other parameters
            device="cuda",
            max_frames=16,
            verbose=True
        )
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test generation
    print(f"\n🎬 Testing video generation...")
    try:
        result = model.generate(
            video_path=VIDEO_PATH,
            question="What is happening in this video?",
            max_new_tokens=50
        )

        print("\n✅ Generation succeeded!")
        print(f"\n📊 Results:")
        print(f"  Answer: {result['generated_text']}")
        print(f"  Frames: {result['num_frames']}")

        # Check Stage 1 statistics
        if 'tokens_before_stage1' in result:
            print(f"\n  Stage 1 Statistics:")
            print(f"    Tokens before: {result['tokens_before_stage1']}")
            print(f"    Tokens after: {result['tokens_after_stage1']}")
            print(f"    Compression: {result['stage1_compression_ratio']:.1%}")
            print(f"    Static tokens: {result['num_static_tokens']}")
            print(f"    Dynamic tokens: {result['num_dynamic_tokens']}")

        return True

    except ValueError as e:
        if "do not match" in str(e):
            print(f"\n❌ Token validation error (expected with V1, should NOT happen with V2):")
            print(f"   {e}")
            return False
        else:
            print(f"\n❌ Unexpected ValueError: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Stage 1 V2 Test - TimeChat-Online Style Token Merging")
    print("="*80)
    print("\nThis test verifies that Stage 1 V2 resolves the token validation issue")
    print("by applying token merging AFTER validation passes.\n")

    success = test_stage1_v2()

    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED - Stage 1 V2 works correctly!")
        print("="*80)
        sys.exit(0)
    else:
        print("❌ TEST FAILED - Stage 1 V2 encountered errors")
        print("="*80)
        sys.exit(1)
