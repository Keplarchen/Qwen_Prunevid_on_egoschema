"""
Qwen2.5-VL + PruneVid主模型封装

提供简单易用的API来使用PruneVid方法对Qwen2.5-VL进行token pruning。

使用示例：
    >>> from qwen_prunevid_new import Qwen25VLPruneVid, PruneVidConfig
    >>>
    >>> config = PruneVidConfig(
    ...     enable_stage1=True,
    ...     enable_pruning=True,
    ...     keep_ratio=0.4,
    ... )
    >>>
    >>> model = Qwen25VLPruneVid(
    ...     model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    ...     config=config,
    ... )
    >>>
    >>> result = model.generate(
    ...     video_path="path/to/video.mp4",
    ...     question="What is happening in the video?",
    ... )
    >>>
    >>> print(result['generated_text'])
    >>> print(f"Tokens reduced: {result['compression_stats']}")
"""

import torch
from typing import Optional, Dict, Union
from pathlib import Path

# 兼容不同版本的 transformers
try:
    from transformers import AutoProcessor
except ImportError:
    try:
        from transformers import Qwen2VLProcessor as AutoProcessor
    except ImportError:
        # 如果都不行，尝试动态加载
        from transformers import AutoTokenizer
        AutoProcessor = AutoTokenizer

from qwen_vl_utils import process_vision_info

# 导入新的 PruneVid 集成模型
try:
    from .modeling_qwen2_5_vl_prunevid_full import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from modeling_qwen2_5_vl_prunevid_full import Qwen2_5_VLForConditionalGeneration

try:
    from .config import PruneVidConfig, get_paper_config
    from .qwen_adapter import Qwen25VLAdapter, create_qwen_adapter
except ImportError:
    from config import PruneVidConfig, get_paper_config
    from qwen_adapter import Qwen25VLAdapter, create_qwen_adapter


class Qwen25VLPruneVid:
    """
    Qwen2.5-VL + PruneVid模型封装

    集成PruneVid的三个阶段，提供简单的generate接口
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[PruneVidConfig] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化模型

        Args:
            model_path: Qwen2.5-VL模型路径或HuggingFace模型ID
            config: PruneVid配置，如果为None则使用论文推荐配置
            device: 计算设备
            torch_dtype: 模型数据类型
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # 加载配置
        if config is None:
            config = get_paper_config()
        self.config = config
        self.config.device = device

        # 加载模型和处理器
        if self.config.verbose:
            print(f"Loading model from {model_path}...")

        # 使用新的 PruneVid 集成模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        if self.config.verbose:
            if self.config.enable_stage1:
                print("✅ Using model with PruneVid Stage 1 integration")
            else:
                print("✅ Using baseline model (Stage 1 disabled)")

        self.processor = AutoProcessor.from_pretrained(model_path)

        # 创建适配器（用于 Stage 2）
        self.adapter = create_qwen_adapter(self.model, self.config)

        if self.config.verbose:
            print("Model loaded successfully!")
            print(self.config)

    def prepare_video_inputs(
        self,
        video_path: str,
        question: str,
    ) -> Dict:
        """
        准备视频输入

        Args:
            video_path: 视频文件路径
            question: 问题文本

        Returns:
            inputs: 处理后的输入字典
        """
        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": self.config.max_pixels,
                        "fps": 1.0,  # 采样率
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 准备输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 保存非tensor字段（如image_grid_thw, video_grid_thw）
        non_tensor_inputs = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                non_tensor_inputs[k] = v

        # 移到设备（只移动tensor）
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        if self.config.verbose:
            print(f"[Debug] Input keys: {list(inputs.keys())}")
            if "image_grid_thw" in inputs:
                print(f"[Debug] image_grid_thw: {inputs['image_grid_thw']}")
            if "video_grid_thw" in inputs:
                print(f"[Debug] video_grid_thw: {inputs['video_grid_thw']}")

        return inputs

    @torch.no_grad()
    def generate(
        self,
        video_path: str,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[str, Dict]:
        """
        生成回答

        Args:
            video_path: 视频文件路径
            question: 问题文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            do_sample: 是否采样
            return_dict: 是否返回详细信息字典

        Returns:
            如果return_dict=True:
                {
                    'generated_text': 生成的文本,
                    'compression_stats': 压缩统计信息,
                    'generation_time': 生成时间（秒）,
                }
            否则:
                生成的文本字符串
        """
        import time

        # 使用配置中的默认值
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if temperature is None:
            temperature = self.config.temperature
        if do_sample is None:
            do_sample = self.config.do_sample

        # 准备输入
        inputs = self.prepare_video_inputs(video_path, question)

        # 获取grid_thw（优先使用video_grid_thw，因为这是视频处理）
        grid_thw = inputs.get("video_grid_thw", inputs.get("image_grid_thw", None))
        if grid_thw is not None and isinstance(grid_thw, torch.Tensor):
            grid_thw = grid_thw.tolist()

        if self.config.verbose:
            print(f"[Debug] Original grid_thw: {grid_thw}")

        # 检测视觉token位置
        visual_start, visual_end, num_visual = self.adapter.detect_visual_token_positions(
            inputs["input_ids"], grid_thw
        )

        # 注册 Stage 2 hooks（如果启用）
        # 注意：Stage 1 已经集成在自定义模型的 forward() 中，不需要 hooks
        if self.config.enable_pruning:
            self.adapter.register_hooks(visual_start, visual_end, grid_thw, inputs)

        if self.config.verbose:
            print(f"\n[Info] Visual tokens: {num_visual} (position {visual_start}:{visual_end})")

        # 创建自定义cache
        past_key_values = self.adapter.create_cache()

        # 生成
        start_time = time.time()

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            past_key_values=past_key_values,
            output_attentions=self.config.enable_pruning,  # Stage 2需要attention
            # PruneVid Stage 1 参数
            enable_stage1=self.config.enable_stage1,
            tau=self.config.tau,
            cluster_ratio=self.config.cluster_ratio,
            temporal_segment_ratio=self.config.temporal_segment_ratio,
            dpc_knn_k=self.config.dpc_knn_k,
            verbose=self.config.verbose,
        )

        generation_time = time.time() - start_time

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # 移除hooks
        self.adapter.remove_hooks()

        # 收集统计信息
        if return_dict:
            stats = self.adapter.get_statistics()

            # 从自定义模型中获取 Stage 1 统计信息（如果使用了自定义模型）
            if self.config.enable_stage1 and hasattr(self.model, 'model') and hasattr(self.model.model, 'stage1_stats'):
                stage1_model_stats = self.model.model.stage1_stats
                if stage1_model_stats:
                    stats["stage1"] = stage1_model_stats

            # 计算总体压缩率
            tokens_before = num_visual
            tokens_after = num_visual

            if self.config.enable_stage1 and "stage1" in stats:
                stage1_stats = stats["stage1"]
                if stage1_stats.get("enabled", False):
                    tokens_after = stage1_stats.get("tokens_after", tokens_after)

            if self.config.enable_pruning and "stage3" in stats:
                stage3_stats = stats["stage3"]
                if stage3_stats.get("compressed", False):
                    tokens_after = stage3_stats.get("kept_visual_tokens", tokens_after)

            compression_ratio = tokens_after / tokens_before if tokens_before > 0 else 1.0

            return {
                "generated_text": generated_text,
                "compression_stats": {
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "compression_ratio": compression_ratio,
                    "reduction_percentage": (1 - compression_ratio) * 100,
                    "detailed_stats": stats,
                },
                "generation_time": generation_time,
                "question": question,
                "video_path": video_path,
            }
        else:
            return generated_text

    def generate_batch(
        self,
        video_paths: list,
        questions: list,
        **kwargs,
    ) -> list:
        """
        批量生成（当前实现为循环调用单个generate）

        Args:
            video_paths: 视频路径列表
            questions: 问题列表
            **kwargs: 传递给generate的其他参数

        Returns:
            results: 结果列表
        """
        results = []
        for video_path, question in zip(video_paths, questions):
            result = self.generate(video_path, question, **kwargs)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            info: 模型信息字典
        """
        return {
            "model_class": self.model.__class__.__name__,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "config": str(self.config),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

    def __repr__(self) -> str:
        return f"Qwen25VLPruneVid(device={self.device}, config={self.config})"


def load_model(
    model_path: str,
    config: Optional[Union[PruneVidConfig, str]] = None,
    device: str = "cuda",
    **kwargs,
) -> Qwen25VLPruneVid:
    """
    加载Qwen2.5-VL + PruneVid模型

    Args:
        model_path: 模型路径
        config: PruneVid配置或预设名称（'baseline', 'paper', 'conservative', 'aggressive'）
        device: 设备
        **kwargs: 传递给Qwen25VLPruneVid的其他参数

    Returns:
        model: Qwen25VLPruneVid实例
    """
    try:
        from .config import (
            get_baseline_config,
            get_paper_config,
            get_conservative_config,
            get_aggressive_config,
        )
    except ImportError:
        from config import (
            get_baseline_config,
            get_paper_config,
            get_conservative_config,
            get_aggressive_config,
        )

    # 处理预设配置
    if isinstance(config, str):
        config_map = {
            "baseline": get_baseline_config,
            "paper": get_paper_config,
            "conservative": get_conservative_config,
            "aggressive": get_aggressive_config,
        }
        if config in config_map:
            config = config_map[config]()
        else:
            raise ValueError(
                f"Unknown config preset: {config}. "
                f"Available: {list(config_map.keys())}"
            )

    return Qwen25VLPruneVid(
        model_path=model_path,
        config=config,
        device=device,
        **kwargs,
    )
