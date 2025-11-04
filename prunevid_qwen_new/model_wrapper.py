"""
PruneVid高层API封装
==================

提供简单易用的接口，用户无需了解内部实现细节即可使用PruneVid。

主要类：
- PruneVidQwen25VL: 统一的高层API

使用示例：
>>> from prunevid_qwen_new import PruneVidQwen25VL, get_paper_config
>>> model = PruneVidQwen25VL("Qwen/Qwen2.5-VL-7B-Instruct", config=get_paper_config())
>>> result = model.generate("video.mp4", "描述视频中发生了什么？")
>>> print(result['answer'])
>>> print(f"Token压缩率: {result['stats']['reduction_percentage']:.1f}%")
"""
import pdb
import torch
from typing import Optional, Dict, Union, List
from pathlib import Path
from PIL import Image
import numpy as np

from config import PruneVidConfig, get_paper_config
from modeling_qwen2_5_vl_prunevid import load_prunevid_model


class PruneVidQwen25VL:
    """
    PruneVid的高层API封装

    这个类提供了最简单的使用方式，自动处理视频加载、推理和结果返回。
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        config: Optional[PruneVidConfig] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化PruneVid模型

        Args:
            model_path: Qwen2.5-VL模型路径或HuggingFace model ID
            config: PruneVid配置，默认使用论文推荐配置
            device: 设备 ("cuda" 或 "cpu")
            torch_dtype: 模型数据类型
        """
        if config is None:
            config = get_paper_config()
            print("使用论文推荐配置 (tau=0.8, cluster_ratio=0.5, keep_ratio=0.4)")

        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype

        # 加载模型和processor
        print(f"正在加载模型: {model_path}")
        self.model, self.processor = load_prunevid_model(
            model_path,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
        )

        print("模型加载完成！")

    def generate(
        self,
        video_path: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        question: str = "请描述视频/图片中的内容。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_stats: bool = True,
        **generate_kwargs
    ) -> Dict:
        """
        对视频或图片生成回答

        Args:
            video_path: 视频文件路径（.mp4, .avi等）
            images: 图片列表（路径或PIL Image对象）
            question: 问题文本
            max_new_tokens: 生成的最大token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否使用采样
            return_stats: 是否返回压缩统计信息
            **generate_kwargs: 其他generate参数

        Returns:
            result: 字典，包含：
                - answer: 生成的回答文本
                - stats: 压缩统计信息（如果return_stats=True）
                - input_tokens: 输入token数量
                - output_tokens: 输出token数量
        """
        # 检查输入
        if video_path is None and images is None:
            raise ValueError("必须提供video_path或images之一")

        if video_path is not None and images is not None:
            raise ValueError("video_path和images不能同时提供")

        # 准备输入
        if video_path is not None:
            # 加载视频
            video_frames = self._load_video(video_path)
            # 构造messages（Qwen2.5-VL格式，添加动态分辨率参数）
            # 关键：使用total_pixels参数控制总像素预算
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames,
                            "min_pixels": self.config.video_min_pixels,
                            "max_pixels": self.config.video_max_pixels,
                            "total_pixels": self.config.video_max_pixels,  # 总像素预算（关键！）
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            # 加载图片
            if isinstance(images, str):
                images = [images]
            image_objects = [
                Image.open(img) if isinstance(img, (str, Path)) else img
                for img in images
            ]
            # 构造messages
            # 注意：像素参数已在processor初始化时设置
            content = []
            for img in image_objects:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": question})

            messages = [{"role": "user", "content": content}]

        # 使用processor处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if video_path is not None:
            # 视频处理（已在_load_video中resize，无需额外参数）
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[video_frames],  # videos参数需要batch格式：list[list[PIL.Image]]
                padding=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.torch_dtype)
        else:
            # 图片处理（像素参数已在messages中设置）
            inputs = self.processor(
                text=[text],
                images=image_objects,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.torch_dtype)

        # 生成
        print(f"开始生成... (输入tokens: {inputs['input_ids'].shape[1]})")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **generate_kwargs
            )

        # 解码
        # 只取新生成的部分
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 构造返回结果
        result = {
            "answer": answer,
            "input_tokens": inputs['input_ids'].shape[1],
            "output_tokens": generated_ids.shape[1],
        }

        # 添加统计信息
        if return_stats and self.config.collect_stats:
            stats = self.model.get_stats()
            result["stats"] = stats

            # 计算总体压缩率
            if stats:
                self._print_stats(stats)

        print(f"\n生成完成！输出tokens: {result['output_tokens']}")

        return result

    def _load_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
        """
        加载视频帧并应用动态分辨率调整

        Args:
            video_path: 视频路径
            max_frames: 最大帧数，默认使用config中的设置

        Returns:
            frames: PIL Image列表（已resize）
        """
        if max_frames is None:
            max_frames = self.config.max_frames

        try:
            import cv2
        except ImportError:
            raise ImportError("需要安装opencv-python: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")

        # 均匀采样
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        # 计算动态分辨率
        # 注意：video_min_pixels和video_max_pixels是总像素预算（T×H×W），不是每帧！
        nframes = len(frame_indices)
        FRAME_FACTOR = 2  # Qwen2.5-VL的帧因子

        # 使用总像素预算上限
        total_pixels_budget = self.config.video_max_pixels

        # 计算每帧可用的像素数：总预算 / 帧数 × FRAME_FACTOR
        max_pixels_per_frame = int(total_pixels_budget / nframes * FRAME_FACTOR)

        print(f"动态分辨率配置:")
        print(f"  - 总预算: {total_pixels_budget:,} 像素 (约 {total_pixels_budget/(32*32):.0f} tokens)")
        print(f"  - 帧数: {nframes}")
        print(f"  - 每帧分配: {max_pixels_per_frame:,} 像素 (约 {max_pixels_per_frame/784:.1f} tokens/帧)")
        print(f"  - 预计总tokens: {nframes * max_pixels_per_frame / 784:.0f}")
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转为PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # 应用smart_resize：保持宽高比，调整到目标像素数，宽高对齐到28
                resized_image = self._smart_resize(pil_image, max_pixels_per_frame, factor=28)
                frames.append(resized_image)

        cap.release()

        print(f"加载了 {len(frames)} 帧 (resize后: {frames[0].size if frames else 'N/A'})")
        return frames

    def _smart_resize(self, image: Image.Image, target_pixels: int, factor: int = 28) -> Image.Image:
        """
        智能resize图片：保持宽高比，调整到目标像素数，宽高对齐到factor的倍数

        这是Qwen2.5-VL的核心resize算法

        Args:
            image: PIL Image
            target_pixels: 目标像素数
            factor: 对齐因子（Qwen2.5-VL使用28）

        Returns:
            resized_image: resize后的PIL Image
        """
        import math

        width, height = image.size
        current_pixels = width * height

        # 计算缩放比例
        if current_pixels > target_pixels:
            # 需要缩小
            scale = math.sqrt(target_pixels / current_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # 不需要缩放或放大（通常视频帧会比target_pixels大）
            new_width = width
            new_height = height

        # 对齐到factor的倍数（向下取整）
        new_width = (new_width // factor) * factor
        new_height = (new_height // factor) * factor

        # 确保至少是1个factor
        new_width = max(factor, new_width)
        new_height = max(factor, new_height)

        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

        return resized

    def _print_stats(self, stats: Dict):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("PruneVid压缩统计")
        print("=" * 50)

        if 'stage1' in stats:
            s1 = stats['stage1']
            print(f"\nStage 1 (时空合并):")
            print(f"  原始tokens: {s1.get('original_tokens', 'N/A')}")
            print(f"  合并后tokens: {s1.get('compressed_tokens', 'N/A')}")
            print(f"  压缩率: {s1.get('reduction_percentage', 0):.1f}%")
            if 'num_segments' in s1:
                print(f"  场景段数: {s1['num_segments']}")
            if 'static_ratio' in s1:
                print(f"  静态token比例: {s1['static_ratio']*100:.1f}%")

        if 'stage2' in stats:
            s2 = stats['stage2']
            print(f"\nStage 2 (注意力选择):")
            print(f"  原始tokens: {s2.get('original_tokens', 'N/A')}")
            print(f"  选择后tokens: {s2.get('compressed_tokens', 'N/A')}")
            print(f"  压缩率: {s2.get('reduction_percentage', 0):.1f}%")

        if 'stage3' in stats:
            s3 = stats['stage3']
            print(f"\nStage 3 (KV缓存压缩):")
            print(f"  原始序列长度: {s3.get('original_seq_len', 'N/A')}")
            print(f"  压缩后序列长度: {s3.get('compressed_seq_len', 'N/A')}")
            print(f"  压缩层数: {s3.get('num_compressed_layers', 'N/A')}")
            print(f"  压缩率: {s3.get('reduction_ratio', 0)*100:.1f}%")

        print("=" * 50)

    def update_config(self, config: PruneVidConfig):
        """
        更新配置

        Args:
            config: 新的PruneVid配置
        """
        self.config = config
        self.model.config = config

        # 重新初始化stages
        if config.enable_stage1:
            from stage1_temporal_spatial_merge import SpatialTemporalTokenMerger
            self.model.stage1 = SpatialTemporalTokenMerger(config)
        else:
            self.model.stage1 = None

        if config.enable_stage2:
            from stage2_attention_selection import AttentionBasedTokenSelector
            self.model.stage2 = AttentionBasedTokenSelector(config)
        else:
            self.model.stage2 = None

        print("配置已更新")

    def __call__(self, *args, **kwargs):
        """使对象可调用，等同于generate方法"""
        return self.generate(*args, **kwargs)


# 导出
__all__ = ["PruneVidQwen25VL"]
