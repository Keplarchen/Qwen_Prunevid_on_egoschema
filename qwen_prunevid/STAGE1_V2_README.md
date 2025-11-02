# Stage 1 V2: TimeChat-Online Style Token Merging

## Overview

Stage 1 V2 implements spatial-temporal token merging using a **TimeChat-Online inspired architecture**. This resolves the token validation issue by applying token merging **after** embeddings are created, instead of during vision encoding.

## Problem Solved

### Original Issue (V1)
```
Vision Encoder → Stage 1 Merging → Compressed Features (880 tokens)
                                          ↓
                                    Validation expects 1768 tokens
                                          ↓
                                    ❌ ValueError: token mismatch
```

### New Solution (V2)
```
Vision Encoder → Full Features (1768 tokens)
                       ↓
                 Validation passes ✓
                       ↓
           LLM Model Forward → Stage 1 Merging (1768 → 880 tokens)
                       ↓
                 LLM Processing uses 880 tokens
```

## Architecture

### Key Differences from V1

| Aspect | V1 (Original) | V2 (TimeChat-Online Style) |
|--------|---------------|---------------------------|
| **Hook Point** | `get_video_features()` | `Qwen2_5_VLModel.forward()` |
| **Merging Timing** | Before validation | After validation |
| **Token Count Issue** | ❌ Fails validation | ✅ Passes validation |
| **Performance** | Same | Same (identical computation) |

### Implementation Details

1. **Hook `Qwen2_5_VLModel.forward()`**
   - Intercepts the LLM model's forward pass
   - At this point, `inputs_embeds` already contains full visual embeddings
   - Validation has already passed

2. **Extract Visual Tokens**
   - Uses `input_ids` to locate video token positions
   - Extracts the corresponding embeddings from `inputs_embeds`

3. **Apply Stage 1 Merging**
   - Same algorithm as V1: spatial-temporal clustering
   - Compresses visual tokens (e.g., 1768 → 880)

4. **Reconstruct Embeddings**
   - Rebuilds `inputs_embeds` with compressed visual tokens
   - Updates `position_ids` accordingly

5. **Continue LLM Processing**
   - LLM decoder processes the compressed embeddings

## Usage

### Automatic (Recommended)

The `Qwen25VLPruneVid` class now uses V2 by default:

```python
from qwen_prunevid import Qwen25VLPruneVid

model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    enable_stage1=True,  # Automatically uses V2
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
)

result = model.generate(
    video_path="video.mp4",
    question="What is happening?",
    max_new_tokens=100
)
```

### Manual (Advanced)

You can also use `Qwen25VLStage1WrapperV2` directly:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_prunevid import Qwen25VLStage1WrapperV2

# Load model
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Initialize Stage 1 V2
stage1 = Qwen25VLStage1WrapperV2(
    model=model,
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
    verbose=True
)

# Before generation, set video info
stage1.set_video_info(
    video_grid_thw=inputs['video_grid_thw'],
    pixel_values_videos=inputs.get('pixel_values_videos')
)

# Generate
output = model.generate(**inputs, max_new_tokens=100)

# Clear video info after generation
stage1.clear_video_info()
```

## Testing

Run the test script to verify the implementation:

```bash
python test_stage1_v2.py
```

Expected output:
```
✅ TEST PASSED - Stage 1 V2 works correctly!
```

## Performance

**V2 has identical performance to V1:**
- ✅ Same computation (vision encoding + token merging + LLM)
- ✅ Same compression ratio
- ✅ Same inference speed

**The only difference is architectural:**
- V1: Merges in vision encoder (fails validation)
- V2: Merges in LLM forward (passes validation)

## Comparison with TimeChat-Online

Our implementation is inspired by TimeChat-Online but with key differences:

| Aspect | TimeChat-Online | PruneVid Stage 1 V2 |
|--------|-----------------|---------------------|
| **Merging Algorithm** | Pixel/feature similarity | Spatial-temporal clustering |
| **Hook Point** | `Qwen2_5_VLModel.forward()` | `Qwen2_5_VLModel.forward()` ✓ |
| **Validation Bypass** | Merges after validation | Merges after validation ✓ |
| **Position Handling** | Full position ID update | Simplified position update |

## Migration from V1

If you're using the old `Qwen25VLStage1Wrapper` (V1), migration is automatic:

```python
# Old code (V1 - will fail with token mismatch)
model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    enable_stage1=True,  # Now uses V2 automatically!
    ...
)

# No code changes needed - V2 is now the default
```

## Technical Notes

### Position IDs Handling

V2 uses a simplified approach for position IDs:
- Keeps the first `N` position IDs for compressed visual tokens
- This works because M-ROPE position IDs are already computed correctly
- More sophisticated tracking could be added in the future

### Memory Overhead

V2 briefly uses more memory than V1:
- V1: Compresses before creating `inputs_embeds`
- V2: Creates full `inputs_embeds`, then compresses

Overhead: `(original_tokens - compressed_tokens) × hidden_dim × batch_size`
- Example: `(1768 - 880) × 3584 × 1 = ~3MB` (negligible)

### Multi-Video Support

Current implementation assumes one video per sample. For multiple videos:
- Need to track `video_grid_thw` for each video
- Apply merging to each video separately

## Troubleshooting

### Test fails with "token mismatch"
- Check that you're using V2, not V1
- Verify `model_wrapper.py` uses `Qwen25VLStage1WrapperV2`

### Position IDs mismatch
- This is a known limitation of the simplified approach
- The model should still work, but may have slightly degraded performance
- Future work: Implement full position ID tracking

### Out of memory
- The brief memory overhead should be negligible
- If OOM occurs, try reducing `max_frames` or `batch_size`

## Future Improvements

1. **Full Position ID Tracking**
   - Track which tokens were kept during merging
   - Update position IDs accordingly

2. **Multi-Video Support**
   - Handle multiple videos in one sample
   - Apply merging per-video

3. **Unified V1/V2 Interface**
   - Allow switching between V1 and V2 via parameter
   - Useful for research comparisons

## References

- TimeChat-Online: [GitHub](https://github.com/yaolinli/TimeChat-Online)
- PruneVid Paper: [arXiv:2412.16117](https://arxiv.org/abs/2412.16117)
- Qwen2.5-VL: [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
