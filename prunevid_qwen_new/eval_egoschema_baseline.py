"""
EgoSchema评估脚本 - 原始Qwen2.5-VL-7B-Instruct (Baseline)

在EgoSchema数据集上评估原始Qwen2.5-VL模型的性能，作为baseline对比。
使用transformers原生实现，不包含任何PruneVid修改。
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import time
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# ============================================================================
# 可调超参数配置区域
# ============================================================================

# GPU配置
GPU_ID = 0  # 使用哪块GPU（0, 1, 2, ...）
DEVICE = f"cuda:{GPU_ID}"

# 数据集配置
VIDEO_DIR = "/mnt/ssd_ext/huggingface/egoschema/videos"  # 视频文件夹
DATASET_NAME = "lmms-lab/egoschema"
DATASET_SPLIT = "test"  # 'test' or 'validation'

# 模型配置
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # Qwen2.5-VL模型路径

# 视频处理配置
MAX_FRAMES = 360  # 最大帧数（与PruneVid版本保持一致）
VIDEO_MIN_PIXELS = 128 * 32 * 32  # 视频最小像素（131,072像素）
VIDEO_MAX_PIXELS = 768 * 32 * 32  # 视频最大像素（786,432像素）

# 生成配置
MAX_NEW_TOKENS = 128  # 最大生成token数
TEMPERATURE = 0.0  # 0表示greedy decoding
DO_SAMPLE = False  # 确定性生成

# 测试配置
NUM_SAMPLES = 500  # 测试样本数量，None表示全部
START_INDEX = 0  # 从第几个样本开始
SAVE_RESULTS = True  # 是否保存结果
OUTPUT_DIR = "./results"  # 结果保存目录
VERBOSE = True  # 是否打印详细信息

# ============================================================================
# 代码开始
# ============================================================================

# 设置CUDA设备
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)


class EgoSchemaBaselineEvaluator:
    """EgoSchema基线评估器 - 使用原始Qwen2.5-VL模型"""

    def __init__(
        self,
        model_path: str,
        video_dir: str,
        dataset_name: str = "lmms-lab/egoschema",
        dataset_split: str = "test",
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.video_dir = Path(video_dir)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.device = device

        # 加载模型和processor
        self.load_model()

        # 加载数据集
        self.load_dataset()

        # 统计信息
        self.total_samples = 0
        self.correct_samples = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.results = []

    def load_model(self):
        """加载原始Qwen2.5-VL模型"""
        print(f"Loading model: {self.model_path}")
        print("This is the BASELINE version using original transformers implementation")

        # 加载模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # 使用FlashAttention2
        ).eval()

        # 加载processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=VIDEO_MIN_PIXELS,
            max_pixels=VIDEO_MAX_PIXELS,
        )

        print("Model loaded successfully!\n")

    def load_dataset(self):
        """加载EgoSchema数据集"""
        print(f"Loading dataset {self.dataset_name} ({self.dataset_split})...")
        self.dataset = load_dataset(self.dataset_name, "Subset", split=self.dataset_split)
        print(f"Loaded {len(self.dataset)} questions\n")

    def load_video(self, video_path: str, max_frames: int = MAX_FRAMES) -> List[Image.Image]:
        """
        加载视频帧

        Args:
            video_path: 视频路径
            max_frames: 最大帧数

        Returns:
            frames: PIL Image列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if VERBOSE:
            print(f"  Video info: {total_frames} frames, {fps:.2f} FPS")

        # 均匀采样
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        # 读取帧
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转为PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()

        if VERBOSE:
            print(f"  Loaded {len(frames)} frames")

        return frames

    def format_question(self, sample: Dict) -> str:
        """
        格式化问题为选择题形式

        Args:
            sample: 标注样本

        Returns:
            formatted_question: 格式化后的问题
        """
        question = sample['question']
        options = sample['option']

        formatted = f"{question}\n\n"
        formatted += "Options:\n"
        for i, opt in enumerate(options):
            formatted += f"{i}. {opt}\n"
        formatted += "\nAnswer with the option number (0-4):"

        return formatted

    def extract_answer(self, generated_text: str) -> Optional[int]:
        """
        从生成的文本中提取答案选项

        Args:
            generated_text: 模型生成的文本

        Returns:
            answer: 答案选项 (0-4)，如果无法提取则返回None
        """
        # 清理文本
        text = generated_text.strip().lower()

        # 尝试多种提取方式
        # 1. 直接是数字
        if text in ['0', '1', '2', '3', '4']:
            return int(text)

        # 2. 字母形式 (A=0, B=1, C=2, D=3, E=4)
        if text.startswith('a') or text.startswith('option a'):
            return 0
        elif text.startswith('b') or text.startswith('option b'):
            return 1
        elif text.startswith('c') or text.startswith('option c'):
            return 2
        elif text.startswith('d') or text.startswith('option d'):
            return 3
        elif text.startswith('e') or text.startswith('option e'):
            return 4

        # 3. 包含"option X"或"选项 X"
        for i in range(5):
            if f"option {i}" in text or f"选项 {i}" in text or f"option{i}" in text:
                return i

        # 4. 查找字母选项
        letter_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        for char in text:
            if char in letter_map:
                return letter_map[char]

        # 5. 查找数字0-4
        for char in text:
            if char in '01234':
                return int(char)

        # 6. 无法提取
        return None

    def evaluate_sample(self, sample: Dict, sample_idx: int) -> Optional[Dict]:
        """
        评估单个样本

        Args:
            sample: 标注样本
            sample_idx: 样本索引

        Returns:
            result: 评估结果
        """
        # 获取视频路径
        video_id = sample['video_idx']
        video_path = self.video_dir / f"{video_id}.mp4"

        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            return None

        # 格式化问题
        question = self.format_question(sample)

        try:
            # 加载视频
            video_frames = self.load_video(str(video_path), max_frames=MAX_FRAMES)

            # 构造messages（Qwen2.5-VL格式）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames,
                            "min_pixels": VIDEO_MIN_PIXELS,
                            "max_pixels": VIDEO_MAX_PIXELS,
                            "total_pixels": VIDEO_MAX_PIXELS,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            # 使用processor处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=None,
                videos=[video_frames],
                padding=True,
                return_tensors="pt",
            ).to(self.device, dtype=torch.bfloat16)

            input_length = inputs['input_ids'].shape[1]
            if VERBOSE:
                print(f"  Input tokens: {input_length}")

            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                )

            # 解码（只取新生成的部分）
            generated_ids = output_ids[:, input_length:]
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            output_length = generated_ids.shape[1]

            # 提取答案
            predicted_answer = self.extract_answer(generated_text)
            ground_truth = int(sample['answer'])

            # 判断正确性
            is_correct = (predicted_answer == ground_truth)

            return {
                'sample_idx': sample_idx,
                'video_id': video_id,
                'question': sample['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'generated_text': generated_text,
                'is_correct': is_correct,
                'input_tokens': input_length,
                'output_tokens': output_length,
            }

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_evaluation(
        self,
        num_samples: Optional[int] = None,
        start_index: int = 0,
    ):
        """
        运行评估

        Args:
            num_samples: 评估样本数量，None表示全部
            start_index: 起始索引
        """
        # 确定要评估的样本
        end_index = len(self.dataset) if num_samples is None else min(start_index + num_samples, len(self.dataset))
        samples_to_eval = self.dataset.select(range(start_index, end_index))

        print(f"\n{'='*80}")
        print(f"Starting BASELINE evaluation on {len(samples_to_eval)} samples (from index {start_index})")
        print(f"Model: {self.model_path}")
        print(f"{'='*80}\n")

        # 评估每个样本
        start_time = time.time()

        for i, sample in enumerate(samples_to_eval):
            sample_idx = start_index + i

            if VERBOSE:
                print(f"Sample {i + 1}/{len(samples_to_eval)}")
                print(f"Video ID: {sample['video_idx']}")
                print(f"Question: {sample['question'][:100]}...")

            # 评估样本
            result = self.evaluate_sample(sample, sample_idx)

            if result is None:
                continue

            # 更新统计
            self.total_samples += 1
            if result['is_correct']:
                self.correct_samples += 1

            self.total_input_tokens += result['input_tokens']
            self.total_output_tokens += result['output_tokens']
            self.results.append(result)

            # 打印结果
            if VERBOSE:
                print(f"\nGround Truth: {result['ground_truth']}")
                print(f"Predicted:    {result['predicted_answer']}")
                print(f"Generated:    '{result['generated_text']}'")
                print(f"Correct: {'✓' if result['is_correct'] else '✗'}")

                # 当前累计准确率
                accuracy = self.correct_samples / self.total_samples * 100
                print(f"\n📊 Current Accuracy: {self.correct_samples}/{self.total_samples} = {accuracy:.2f}%")

                # 平均token数量
                avg_input = self.total_input_tokens / self.total_samples
                avg_output = self.total_output_tokens / self.total_samples
                print(f"📊 Average tokens - Input: {avg_input:.0f}, Output: {avg_output:.0f}")
                print(f"\n{'='*80}\n")

        total_time = time.time() - start_time

        # 打印最终结果
        self.print_final_results(total_time)

    def print_final_results(self, total_time: float):
        """打印最终评估结果"""
        print(f"\n{'='*80}")
        print(f"🎉 BASELINE EVALUATION COMPLETED")
        print(f"{'='*80}\n")

        # 准确率
        accuracy = self.correct_samples / self.total_samples * 100 if self.total_samples > 0 else 0
        print(f"📊 Final Accuracy:")
        print(f"  Correct: {self.correct_samples}/{self.total_samples}")
        print(f"  Accuracy: {accuracy:.2f}%")

        # Token统计
        avg_input = self.total_input_tokens / self.total_samples if self.total_samples > 0 else 0
        avg_output = self.total_output_tokens / self.total_samples if self.total_samples > 0 else 0
        print(f"\n📊 Token Statistics:")
        print(f"  Total input tokens:  {self.total_input_tokens}")
        print(f"  Total output tokens: {self.total_output_tokens}")
        print(f"  Avg input tokens:    {avg_input:.1f}")
        print(f"  Avg output tokens:   {avg_output:.1f}")

        # 时间统计
        avg_time_per_sample = total_time / self.total_samples if self.total_samples > 0 else 0
        print(f"\n⏱️  Time Statistics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per sample: {avg_time_per_sample:.2f}s")

        print(f"\n{'='*80}\n")

        # 保存结果
        if SAVE_RESULTS:
            self.save_results()

    def save_results(self):
        """保存评估结果"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"egoschema_results_{timestamp}_baseline.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 准备保存的数据
        save_data = {
            'config': {
                'model_path': self.model_path,
                'model_type': 'baseline',
                'max_frames': MAX_FRAMES,
                'video_min_pixels': VIDEO_MIN_PIXELS,
                'video_max_pixels': VIDEO_MAX_PIXELS,
                'num_samples': NUM_SAMPLES,
                'dataset': self.dataset_name,
                'split': self.dataset_split,
            },
            'summary': {
                'total_samples': self.total_samples,
                'correct_samples': self.correct_samples,
                'accuracy': self.correct_samples / self.total_samples * 100 if self.total_samples > 0 else 0,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'avg_input_tokens': self.total_input_tokens / self.total_samples if self.total_samples > 0 else 0,
                'avg_output_tokens': self.total_output_tokens / self.total_samples if self.total_samples > 0 else 0,
            },
            'results': self.results,
        }

        # 保存
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"✅ Results saved to: {filepath}")


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"EgoSchema Evaluation - Qwen2.5-VL Baseline (No PruneVid)")
    print(f"{'='*80}\n")

    print(f"Configuration:")
    print(f"  GPU: {GPU_ID} (device: {DEVICE})")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Dataset: {DATASET_NAME} ({DATASET_SPLIT})")
    print(f"  Video dir: {VIDEO_DIR}")
    print(f"\nVideo Settings:")
    print(f"  Max frames: {MAX_FRAMES}")
    print(f"  Video min pixels: {VIDEO_MIN_PIXELS:,}")
    print(f"  Video max pixels: {VIDEO_MAX_PIXELS:,}")
    print(f"\nTest Settings:")
    print(f"  Num samples: {NUM_SAMPLES if NUM_SAMPLES else 'All'}")
    print(f"  Start index: {START_INDEX}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Do sample: {DO_SAMPLE}")
    print(f"\n{'='*80}\n")

    # 创建评估器
    evaluator = EgoSchemaBaselineEvaluator(
        model_path=MODEL_PATH,
        video_dir=VIDEO_DIR,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        device=DEVICE,
    )

    # 运行评估
    evaluator.run_evaluation(
        num_samples=NUM_SAMPLES,
        start_index=START_INDEX,
    )


if __name__ == "__main__":
    main()
