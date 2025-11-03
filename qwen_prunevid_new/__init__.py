"""
PruneVid: Visual Token Pruning for Efficient Video Large Language Models
Implementation for Qwen2.5-VL

Based on the paper:
"PruneVid: Visual Token Pruning for Efficient Video Large Language Models"
ACL 2025 Findings
"""

from .config import PruneVidConfig
from .model_wrapper import Qwen25VLPruneVid

__version__ = "1.0.0"
__all__ = ["PruneVidConfig", "Qwen25VLPruneVid"]
