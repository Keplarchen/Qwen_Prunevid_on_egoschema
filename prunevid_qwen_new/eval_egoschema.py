"""
EgoSchema评估脚本 - PruneVid for Qwen2.5-VL (新实现)

在EgoSchema数据集上评估PruneVid方法的性能和token压缩效果
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import time
from datasets import load_dataset

# ============================================================================
# 可调超参数配置区域
# ============================================================================

# GPU配置
GPU_ID = 2 # 使用哪块GPU（0, 1, 2, ...）
DEVICE = f"cuda:{GPU_ID}"

# 数据集配置
VIDEO_DIR = "/mnt/ssd_ext/huggingface/egoschema/videos"  # 视频文件夹
# 使用datasets库加载EgoSchema数据集
DATASET_NAME = "lmms-lab/egoschema"
DATASET_SPLIT = "test"  # 'test' or 'validation'

# 模型配置
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # Qwen2.5-VL模型路径（可以是HF ID或本地路径）

# PruneVid配置模式
CONFIG_MODE = "custom"  # 可选: "baseline", "paper", "conservative", "aggressive", "custom"

# 自定义配置（仅当CONFIG_MODE="custom"时使用）
# Stage 1: 时空Token合并
CUSTOM_ENABLE_STAGE1 = False
CUSTOM_TAU = 0.8  # 静态/动态分离阈值 (0.6-0.9)
CUSTOM_CLUSTER_RATIO = 0.5  # 空间聚类保留比例 (0.3-0.7)
CUSTOM_TEMPORAL_SEGMENT_RATIO = 0.25  # 时序分段比例 (0.125-0.5)
CUSTOM_DPC_KNN_K = 5  # DPC-KNN的k近邻参数

# Stage 2: 基于注意力的Token选择
CUSTOM_ENABLE_STAGE2 = False   
CUSTOM_KEEP_RATIO = 0.3  # Token保留比例 (0.2-0.6)
CUSTOM_PRUNING_LAYER = 10  # 在哪一层进行剪枝 (5-15)
CUSTOM_ATTENTION_AGGREGATION = "max"  # 'max' or 'mean'

# Stage 3: KV缓存压缩
CUSTOM_ENABLE_CACHE_COMPRESSION = False

# 视频处理配置
MAX_FRAMES = 16  # 最大帧数 (8, 16, 32)

# 生成配置
MAX_NEW_TOKENS = 10  # 给模型充分空间解释答案（去掉限制）
TEMPERATURE = 0.0  # 0表示greedy decoding
DO_SAMPLE = False

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

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PruneVidConfig,
    get_baseline_config,
    get_paper_config,
    get_conservative_config,
    get_aggressive_config
)
from model_wrapper import PruneVidQwen25VL


class EgoSchemaEvaluator:
    """EgoSchema评估器"""

    def __init__(
        self,
        model: PruneVidQwen25VL,
        video_dir: str,
        dataset_name: str = "lmms-lab/egoschema",
        dataset_split: str = "test",
    ):
        self.model = model
        self.video_dir = Path(video_dir)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        # 加载数据集
        self.load_dataset()

        # 统计信息
        self.total_samples = 0
        self.correct_samples = 0
        self.total_tokens_before = 0
        self.total_tokens_after_stage1 = 0
        self.total_tokens_after_stage2 = 0
        self.total_tokens_after = 0
        self.results = []

    def load_dataset(self):
        """加载EgoSchema数据集"""
        print(f"Loading dataset {self.dataset_name} ({self.dataset_split})...")
        self.dataset = load_dataset(self.dataset_name, "Subset", split=self.dataset_split)
        print(f"Loaded {len(self.dataset)} questions")

    def format_question(self, sample: Dict) -> str:
        """
        格式化问题为选择题形式

        Args:
            sample: 标注样本

        Returns:
            formatted_question: 格式化后的问题
        """
        question = sample['question']
        options = sample['option']  # options is already a list

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

        # 模型推理
        try:
            result = self.model.generate(
                video_path=str(video_path),
                question=question,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                return_stats=True,
            )

            generated_text = result['answer']
            stats = result.get('stats', {})

            # 提取答案
            predicted_answer = self.extract_answer(generated_text)
            ground_truth = int(sample['answer'])

            # 判断正确性
            is_correct = (predicted_answer == ground_truth)

            # 获取token统计
            # 从新实现的stats格式中提取信息
            stage1_stats = stats.get('stage1', {})
            stage2_stats = stats.get('stage2', {})

            # 获取原始token数量
            if stage1_stats:
                tokens_before = stage1_stats.get('original_tokens', 0)
                tokens_after_stage1 = stage1_stats.get('compressed_tokens', tokens_before)
            else:
                # Stage 1禁用，使用input_tokens作为原始值
                tokens_before = result.get('input_tokens', 0)
                tokens_after_stage1 = tokens_before

            # 获取Stage 2后的token数量
            if stage2_stats:
                # Stage 2启用，使用其压缩后的值
                tokens_after_stage2 = stage2_stats.get('compressed_tokens', tokens_after_stage1)
            else:
                # Stage 2禁用，保持Stage 1的值
                tokens_after_stage2 = tokens_after_stage1

            # 返回结果
            return {
                'sample_idx': sample_idx,
                'video_id': video_id,
                'question': sample['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'generated_text': generated_text,
                'is_correct': is_correct,
                'tokens_before': tokens_before,
                'tokens_after_stage1': tokens_after_stage1,
                'tokens_after_stage2': tokens_after_stage2,
                'tokens_after': tokens_after_stage2,
                'stats': stats,
            }

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_sample_result(self, result: Dict, total_samples: int):
        """打印单个样本的结果"""
        # 打印样本信息（问题编号、视频ID、问题）
        print(f"Sample {result['sample_idx'] + 1}/{total_samples}")
        print(f"Video ID: {result['video_id']}")
        print(f"Question: {result['question'][:100]}...")
        print()

        # 详细处理过程和压缩统计会在这里自动打印（由模型的verbose输出）

        # 打印预测结果
        print(f"\nGround Truth: {result['ground_truth']}")
        print(f"Predicted:    {result['predicted_answer']}")
        print(f"Correct: {'✓' if result['is_correct'] else '✗'}")

        # 当前累计准确率
        accuracy = self.correct_samples / self.total_samples * 100
        print(f"\n📊 Current Accuracy: {self.correct_samples}/{self.total_samples} = {accuracy:.2f}%")

        # Token压缩统计
        tokens_before = result['tokens_before']
        tokens_after_stage1 = result['tokens_after_stage1']
        tokens_after_stage2 = result['tokens_after_stage2']

        stage1_drop = (tokens_before - tokens_after_stage1) / tokens_before * 100 if tokens_before > 0 else 0
        stage2_drop = (tokens_after_stage1 - tokens_after_stage2) / tokens_after_stage1 * 100 if tokens_after_stage1 > 0 else 0
        total_drop = (tokens_before - tokens_after_stage2) / tokens_before * 100 if tokens_before > 0 else 0

        print(f"\n📉 Token Compression (Current Sample):")
        print(f"  Original:      {tokens_before}")
        print(f"  After Stage 1: {tokens_after_stage1} (drop: {stage1_drop:.1f}%)")
        print(f"  After Stage 2: {tokens_after_stage2} (drop: {stage2_drop:.1f}%)")
        print(f"  Total drop:    {total_drop:.1f}%")

        # 累计平均压缩率
        avg_stage1_drop = (self.total_tokens_before - self.total_tokens_after_stage1) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0
        avg_stage2_drop = (self.total_tokens_after_stage1 - self.total_tokens_after_stage2) / self.total_tokens_after_stage1 * 100 if self.total_tokens_after_stage1 > 0 else 0
        avg_total_drop = (self.total_tokens_before - self.total_tokens_after_stage2) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0

        print(f"\n📉 Average Token Compression (Cumulative):")
        print(f"  Stage 1 avg drop: {avg_stage1_drop:.1f}%")
        print(f"  Stage 2 avg drop: {avg_stage2_drop:.1f}%")
        print(f"  Total avg drop:   {avg_total_drop:.1f}%")

        # 问题之间的分割线
        print(f"\n{'='*80}\n")

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
        print(f"Starting evaluation on {len(samples_to_eval)} samples (from index {start_index})")
        print(f"{'='*80}\n")

        # 评估每个样本
        start_time = time.time()

        for i, sample in enumerate(samples_to_eval):
            sample_idx = start_index + i
            total_samples_count = len(samples_to_eval)

            # 在评估前打印样本头信息
            if VERBOSE:
                print(f"Sample {i + 1}/{total_samples_count}")
                print(f"Video ID: {sample['video_idx']}")
                print(f"Question: {sample['question'][:100]}...")
                print()

            # 评估样本
            result = self.evaluate_sample(sample, sample_idx)

            if result is None:
                continue

            # 更新统计
            self.total_samples += 1
            if result['is_correct']:
                self.correct_samples += 1

            self.total_tokens_before += result['tokens_before']
            self.total_tokens_after_stage1 += result['tokens_after_stage1']
            self.total_tokens_after_stage2 += result['tokens_after_stage2']
            self.total_tokens_after += result['tokens_after']

            self.results.append(result)

            # 打印结果（不再打印样本头信息，因为已经在前面打印了）
            if VERBOSE:
                # 打印预测结果和统计信息
                print(f"\nGround Truth: {result['ground_truth']}")
                print(f"Predicted:    {result['predicted_answer']}")
                print(f"Generated:    '{result['generated_text']}'")  # 显示实际生成的文本
                print(f"Correct: {'✓' if result['is_correct'] else '✗'}")

                # 当前累计准确率
                accuracy = self.correct_samples / self.total_samples * 100
                print(f"\n📊 Current Accuracy: {self.correct_samples}/{self.total_samples} = {accuracy:.2f}%")

                # Token压缩统计
                tokens_before = result['tokens_before']
                tokens_after_stage1 = result['tokens_after_stage1']
                tokens_after_stage2 = result['tokens_after_stage2']

                stage1_drop = (tokens_before - tokens_after_stage1) / tokens_before * 100 if tokens_before > 0 else 0
                stage2_drop = (tokens_after_stage1 - tokens_after_stage2) / tokens_after_stage1 * 100 if tokens_after_stage1 > 0 else 0
                total_drop = (tokens_before - tokens_after_stage2) / tokens_before * 100 if tokens_before > 0 else 0

                print(f"\n📉 Token Compression (Current Sample):")
                print(f"  Original:      {tokens_before}")
                print(f"  After Stage 1: {tokens_after_stage1} (drop: {stage1_drop:.1f}%)")
                print(f"  After Stage 2: {tokens_after_stage2} (drop: {stage2_drop:.1f}%)")
                print(f"  Total drop:    {total_drop:.1f}%")

                # 累计平均压缩率
                avg_stage1_drop = (self.total_tokens_before - self.total_tokens_after_stage1) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0
                avg_stage2_drop = (self.total_tokens_after_stage1 - self.total_tokens_after_stage2) / self.total_tokens_after_stage1 * 100 if self.total_tokens_after_stage1 > 0 else 0
                avg_total_drop = (self.total_tokens_before - self.total_tokens_after_stage2) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0

                print(f"\n📉 Average Token Compression (Cumulative):")
                print(f"  Stage 1 avg drop: {avg_stage1_drop:.1f}%")
                print(f"  Stage 2 avg drop: {avg_stage2_drop:.1f}%")
                print(f"  Total avg drop:   {avg_total_drop:.1f}%")

                # 问题之间的分割线
                print(f"\n{'='*80}\n")

        total_time = time.time() - start_time

        # 打印最终结果
        self.print_final_results(total_time)

    def print_final_results(self, total_time: float):
        """打印最终评估结果"""
        print(f"\n{'='*80}")
        print(f"🎉 EVALUATION COMPLETED")
        print(f"{'='*80}\n")

        # 准确率
        accuracy = self.correct_samples / self.total_samples * 100 if self.total_samples > 0 else 0
        print(f"📊 Final Accuracy:")
        print(f"  Correct: {self.correct_samples}/{self.total_samples}")
        print(f"  Accuracy: {accuracy:.2f}%")

        # Token压缩统计
        avg_stage1_drop = (self.total_tokens_before - self.total_tokens_after_stage1) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0
        avg_stage2_drop = (self.total_tokens_after_stage1 - self.total_tokens_after_stage2) / self.total_tokens_after_stage1 * 100 if self.total_tokens_after_stage1 > 0 else 0
        avg_total_drop = (self.total_tokens_before - self.total_tokens_after_stage2) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0

        print(f"\n📉 Final Token Compression:")
        print(f"  Total tokens before:       {self.total_tokens_before}")
        print(f"  Total tokens after Stage 1: {self.total_tokens_after_stage1}")
        print(f"  Total tokens after Stage 2: {self.total_tokens_after_stage2}")
        print(f"\n  Stage 1 drop ratio: {avg_stage1_drop:.2f}%")
        print(f"  Stage 2 drop ratio: {avg_stage2_drop:.2f}%")
        print(f"  Total drop ratio:   {avg_total_drop:.2f}%")

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
        filename = f"egoschema_results_{timestamp}_{CONFIG_MODE}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 准备保存的数据
        save_data = {
            'config': {
                'config_mode': CONFIG_MODE,
                'model_config': self.model.config.to_dict(),
                'max_frames': MAX_FRAMES,
                'num_samples': NUM_SAMPLES,
            },
            'summary': {
                'total_samples': self.total_samples,
                'correct_samples': self.correct_samples,
                'accuracy': self.correct_samples / self.total_samples * 100 if self.total_samples > 0 else 0,
                'stage1_drop_ratio': (self.total_tokens_before - self.total_tokens_after_stage1) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0,
                'stage2_drop_ratio': (self.total_tokens_after_stage1 - self.total_tokens_after_stage2) / self.total_tokens_after_stage1 * 100 if self.total_tokens_after_stage1 > 0 else 0,
                'total_drop_ratio': (self.total_tokens_before - self.total_tokens_after_stage2) / self.total_tokens_before * 100 if self.total_tokens_before > 0 else 0,
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
    print(f"EgoSchema Evaluation - PruneVid for Qwen2.5-VL (New Implementation)")
    print(f"{'='*80}\n")

    # 创建配置
    if CONFIG_MODE == "baseline":
        config = get_baseline_config()
        print("Using BASELINE config (no pruning)")
    elif CONFIG_MODE == "paper":
        config = get_paper_config()
        print("Using PAPER config (tau=0.8, keep_ratio=0.4)")
    elif CONFIG_MODE == "conservative":
        config = get_conservative_config()
        print("Using CONSERVATIVE config (high compression)")
    elif CONFIG_MODE == "aggressive":
        config = get_aggressive_config()
        print("Using AGGRESSIVE config (low compression)")
    elif CONFIG_MODE == "custom":
        config = PruneVidConfig(
            # Stage 1
            enable_stage1=CUSTOM_ENABLE_STAGE1,
            tau=CUSTOM_TAU,
            cluster_ratio=CUSTOM_CLUSTER_RATIO,
            temporal_segment_ratio=CUSTOM_TEMPORAL_SEGMENT_RATIO,
            dpc_knn_k=CUSTOM_DPC_KNN_K,
            # Stage 2
            enable_stage2=CUSTOM_ENABLE_STAGE2,
            keep_ratio=CUSTOM_KEEP_RATIO,
            pruning_layer=CUSTOM_PRUNING_LAYER,
            attention_aggregation=CUSTOM_ATTENTION_AGGREGATION,
            # Stage 3
            enable_cache_compression=CUSTOM_ENABLE_CACHE_COMPRESSION,
            # Video
            max_frames=MAX_FRAMES,
            # Debug
            verbose=VERBOSE,
            collect_stats=True,
        )
        print("Using CUSTOM config")
    else:
        raise ValueError(f"Unknown CONFIG_MODE: {CONFIG_MODE}")

    # 更新max_frames
    config.max_frames = MAX_FRAMES

    # 打印配置
    print(f"\nConfiguration:")
    print(f"  GPU: {GPU_ID} (device: {DEVICE})")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Dataset: {DATASET_NAME} ({DATASET_SPLIT})")
    print(f"  Video dir: {VIDEO_DIR}")
    print(f"\nPruneVid Settings:")
    print(f"  Stage 1: {'ON' if config.enable_stage1 else 'OFF'}")
    if config.enable_stage1:
        print(f"    - tau: {config.tau}")
        print(f"    - cluster_ratio: {config.cluster_ratio}")
        print(f"    - temporal_segment_ratio: {config.temporal_segment_ratio}")
    print(f"  Stage 2: {'ON' if config.enable_stage2 else 'OFF'}")
    if config.enable_stage2:
        print(f"    - keep_ratio: {config.keep_ratio}")
        print(f"    - pruning_layer: {config.pruning_layer}")
    print(f"  Stage 3: {'ON' if config.enable_cache_compression else 'OFF'}")
    print(f"\nTest Settings:")
    print(f"  Num samples: {NUM_SAMPLES if NUM_SAMPLES else 'All'}")
    print(f"  Start index: {START_INDEX}")
    print(f"  Max frames: {MAX_FRAMES}")
    print(f"\n{'='*80}\n")

    # 加载模型
    print("Loading model...")
    model = PruneVidQwen25VL(
        model_path=MODEL_PATH,
        config=config,
        device=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded!\n")

    # 创建评估器
    evaluator = EgoSchemaEvaluator(
        model=model,
        video_dir=VIDEO_DIR,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
    )

    # 运行评估
    evaluator.run_evaluation(
        num_samples=NUM_SAMPLES,
        start_index=START_INDEX,
    )


if __name__ == "__main__":
    main()
