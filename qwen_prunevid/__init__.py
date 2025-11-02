"""
PruneVid for Qwen2.5-VL

Visual Token Pruning for Efficient Video Large Language Models
Based on the ACL 2025 paper: "PruneVid: Visual Token Pruning for Efficient Video Large Language Models"

Three-stage pruning approach:
    - Stage 1: Spatial-Temporal Token Merging (Vision Encoder)
    - Stage 2: Attention-Based Token Pruning (LLM)
    - Stage 3: KV Cache Compression (LLM)

Usage:
    from qwen_prunevid import Qwen25VLPruneVid

    # Full three-stage PruneVid
    model = Qwen25VLPruneVid(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        enable_stage1=True,
        tau=0.8,
        cluster_ratio=0.5,
        temporal_segment_ratio=0.25,
        enable_pruning=True,
        keep_ratio=0.4,
        pruning_layer=10,
    )

    result = model.generate(
        video_path="video.mp4",
        question="What is happening in this video?",
        max_new_tokens=100
    )

    print(f"Compression: {result['total_compression_ratio']:.1%}")
"""

from .model_wrapper import Qwen25VLPruneVid
from .prunevid_core import PruneVidCore
from .qwen25_adapter import Qwen25VLAdapter
from .stage1_wrapper import Qwen25VLStage1Wrapper
from .stage1_wrapper_v2 import Qwen25VLStage1WrapperV2
from .stage1_utils import (
    cluster_dpc_knn,
    refine_clusters,
    segment_lengths,
    compute_cluster_vectors,
    index_points
)

__version__ = "2.1.0"  # Updated to 2.1.0 with Stage 1 V2 (TimeChat-Online style)
__all__ = [
    "Qwen25VLPruneVid",
    "PruneVidCore",
    "Qwen25VLAdapter",
    "Qwen25VLStage1Wrapper",
    "Qwen25VLStage1WrapperV2",
    "cluster_dpc_knn",
    "refine_clusters",
    "segment_lengths",
    "compute_cluster_vectors",
    "index_points"
]
