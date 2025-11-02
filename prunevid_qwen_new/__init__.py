"""
PruneVid for Qwen2.5-VL
=======================

这个包实现了PruneVid视觉token剪枝方法在Qwen2.5-VL模型上的应用。

PruneVid通过三个阶段实现高效的视频理解：
1. Stage 1: 时空Token合并 - 减少视频固有冗余
2. Stage 2: 基于注意力的Token选择 - 保留与问题相关的视觉token
3. Stage 3: KV缓存压缩 - 减少生成阶段的内存和计算

主要类：
- PruneVidConfig: 配置类
- PruneVidQwen25VL: 高层API封装
- Qwen2VLForConditionalGenerationWithPruneVid: 集成模型

快速开始：
>>> from prunevid_qwen_new import PruneVidQwen25VL, get_paper_config
>>> model = PruneVidQwen25VL("Qwen/Qwen2.5-VL-7B-Instruct", config=get_paper_config())
>>> result = model.generate("video.mp4", "描述视频内容")
>>> print(result['answer'])

参考论文：
PruneVid: Visual Token Pruning for Efficient Video Large Language Models
ACL 2025 Findings
"""

from .config import PruneVidConfig, get_baseline_config, get_paper_config, get_conservative_config, get_aggressive_config
from .model_wrapper import PruneVidQwen25VL

__version__ = "1.0.0"
__author__ = "PruneVid Implementation for Qwen2.5-VL"

__all__ = [
    "PruneVidConfig",
    "PruneVidQwen25VL",
    "get_baseline_config",
    "get_paper_config",
    "get_conservative_config",
    "get_aggressive_config",
]
