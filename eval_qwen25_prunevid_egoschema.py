#!/usr/bin/env python3
"""
Evaluation Script: Qwen2.5-VL + PruneVid on EgoSchema

Based on the ACL 2025 paper: "PruneVid: Visual Token Pruning for Efficient Video Large Language Models"
Paper: https://arxiv.org/abs/2412.16117
Official Code: https://github.com/Visual-AI/PruneVid
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from qwen_prunevid import Qwen25VLPruneVid


# ============================================================================
# 配置参数 - 基于 PruneVid 论文的标准设置
# ============================================================================
class Config:
    """
    实验配置 - 所有参数集中管理

    PruneVid 论文默认值（三阶段）:
    - Stage 1: tau=0.8, cluster_ratio=0.5, temporal_segment_ratio=0.25
    - Stage 2: keep_ratio=0.4, pruning_layer=10
    - max_frames: 16 (PLLaVA/ST-LLM) 或 32 (LLaVA-OneVision)
    """

    # ===== 模型配置 =====
    MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"
    GPU_ID = 0  # GPU 编号，None = CPU

    # ===== 视频采样配置 =====
    MAX_FRAMES = 16  # 最大帧数 (推荐: 16)
    MIN_FRAMES = 4   # 最小帧数
    FPS = None       # None = 均匀采样，float = 按FPS采样

    # ===== Stage 1: Spatial-Temporal Token Merging =====
    ENABLE_STAGE1 = True              # 启用Stage 1 (Vision Encoder层面)
    TAU = 0.1                          # 静态/动态阈值 [0.7-0.9]
    CLUSTER_RATIO = 0.5                # 空间聚类保留比例 [0.3-0.8]
    TEMPORAL_SEGMENT_RATIO = 0.25      # 时序分段比例 [0.25-0.5]

    # ===== Stage 2: Attention-Based Token Pruning =====
    ENABLE_PRUNING = True   # 启用Stage 2 (LLM层面)
    KEEP_RATIO = 0.5        # Token保留比例 [0.3-0.6] (论文默认值)
    PRUNING_LAYER = 10      # 在第10层剪枝 (论文默认值)

    # ===== 数据集配置 =====
    DATASET_PATH = "datasets--lmms-lab--egoschema/snapshots/58350524ea7eb29c47000121f4f4b65eb6b4acb9/Subset/test-00000-of-00001.parquet"
    VIDEO_DIR = "egoschema/videos"
    NUM_SAMPLES = 10   # 测试样本数量
    START_IDX = 0      # 起始索引

    # ===== 输出配置 =====
    OUTPUT_DIR = "results"
    EXP_NAME = None  # 自动生成
    VERBOSE = False  # 详细调试输出

    # ===== 其他 =====
    SAVE_INTERVAL = 10  # 每N个样本保存一次


# ============================================================================
# 工具函数
# ============================================================================

def convert_to_serializable(obj):
    """Convert numpy/torch types to Python types for JSON."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def print_config(cfg):
    """Print experiment configuration."""
    print("=" * 80)
    print("实验配置 - PruneVid + Qwen2.5-VL on EgoSchema")
    print("=" * 80)
    print(f"\n📦 模型:")
    print(f"  路径: {cfg.MODEL_PATH}")
    print(f"  设备: GPU {cfg.GPU_ID}" if cfg.GPU_ID is not None else "  设备: CPU")

    print(f"\n🎬 视频采样:")
    print(f"  最大帧数: {cfg.MAX_FRAMES}")
    print(f"  最小帧数: {cfg.MIN_FRAMES}")
    print(f"  采样方式: {'FPS=' + str(cfg.FPS) if cfg.FPS else '均匀采样'}")

    print(f"\n✂️  PruneVid (三阶段):")
    print(f"  Stage 1 (时空Token合并): {'启用' if cfg.ENABLE_STAGE1 else '禁用'}")
    if cfg.ENABLE_STAGE1:
        print(f"    tau: {cfg.TAU}")
        print(f"    cluster_ratio: {cfg.CLUSTER_RATIO}")
        print(f"    temporal_segment_ratio: {cfg.TEMPORAL_SEGMENT_RATIO}")

    print(f"  Stage 2 (Attention剪枝): {'启用' if cfg.ENABLE_PRUNING else '禁用'}")
    if cfg.ENABLE_PRUNING:
        print(f"    keep_ratio: {cfg.KEEP_RATIO} (删除 {1-cfg.KEEP_RATIO:.1%})")
        print(f"    pruning_layer: Layer {cfg.PRUNING_LAYER}")

    print(f"\n📊 数据集:")
    print(f"  测试样本: {cfg.NUM_SAMPLES} (索引 {cfg.START_IDX} - {cfg.START_IDX + cfg.NUM_SAMPLES - 1})")

    print(f"\n💾 输出:")
    # 生成实验名称
    if cfg.EXP_NAME:
        exp_name = cfg.EXP_NAME
    else:
        parts = []
        if cfg.ENABLE_STAGE1:
            parts.append(f"s1_tau{cfg.TAU}_c{cfg.CLUSTER_RATIO}")
        if cfg.ENABLE_PRUNING:
            parts.append(f"s2_k{cfg.KEEP_RATIO}_l{cfg.PRUNING_LAYER}")
        if not parts:
            exp_name = f"baseline_f{cfg.MAX_FRAMES}"
        else:
            exp_name = "_".join(parts) + f"_f{cfg.MAX_FRAMES}"
    print(f"  实验名称: {exp_name}")
    print(f"  输出目录: {cfg.OUTPUT_DIR}/{exp_name}")

    print("=" * 80 + "\n")


# ============================================================================
# 主评估函数
# ============================================================================

def main():
    """主函数"""
    cfg = Config()

    # 设置 GPU
    if cfg.GPU_ID is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
        device = "cuda"
    else:
        device = "cpu"

    # 打印配置
    print_config(cfg)

    # 生成实验名称
    if cfg.EXP_NAME is None:
        parts = []
        if cfg.ENABLE_STAGE1:
            parts.append(f"s1_tau{cfg.TAU}_c{cfg.CLUSTER_RATIO}_t{cfg.TEMPORAL_SEGMENT_RATIO}")
        if cfg.ENABLE_PRUNING:
            parts.append(f"s2_k{cfg.KEEP_RATIO}_l{cfg.PRUNING_LAYER}")
        if not parts:
            cfg.EXP_NAME = f"baseline_f{cfg.MAX_FRAMES}"
        else:
            cfg.EXP_NAME = "_".join(parts) + f"_f{cfg.MAX_FRAMES}"

    # 创建输出目录
    output_dir = Path(cfg.OUTPUT_DIR) / cfg.EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}\n")

    # 保存配置
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 加载数据集
    print(f"📥 加载数据集...")
    df = pd.read_parquet(cfg.DATASET_PATH)
    end_idx = min(cfg.START_IDX + cfg.NUM_SAMPLES, len(df))
    df = df.iloc[cfg.START_IDX:end_idx]
    print(f"   样本范围: {cfg.START_IDX} - {end_idx - 1} (共 {len(df)} 个)\n")

    # 初始化模型
    print(f"🚀 初始化模型...")
    model = Qwen25VLPruneVid(
        model_path=cfg.MODEL_PATH,
        # Stage 1参数
        enable_stage1=cfg.ENABLE_STAGE1,
        tau=cfg.TAU,
        cluster_ratio=cfg.CLUSTER_RATIO,
        temporal_segment_ratio=cfg.TEMPORAL_SEGMENT_RATIO,
        # Stage 2参数
        enable_pruning=cfg.ENABLE_PRUNING,
        keep_ratio=cfg.KEEP_RATIO,
        pruning_layer=cfg.PRUNING_LAYER,
        # 其他参数
        device=device,
        max_frames=cfg.MAX_FRAMES,
        min_frames=cfg.MIN_FRAMES,
        fps=cfg.FPS,
        verbose=cfg.VERBOSE
    )
    print()

    # 开始评估
    print("=" * 80)
    print("开始评估")
    print("=" * 80 + "\n")

    results = []
    total_correct = 0
    total_answered = 0
    all_stage1_ratios = []
    all_stage2_ratios = []
    all_total_ratios = []

    for idx, row in df.iterrows():
        sample_idx = idx - cfg.START_IDX
        video_path = os.path.join(cfg.VIDEO_DIR, f"{row['video_idx']}.mp4")

        print(f"[{sample_idx + 1}/{len(df)}] Video: {row['video_idx']}")
        print("-" * 80)

        # 检查视频文件
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}\n")
            results.append({
                'sample_id': cfg.START_IDX + sample_idx,
                'video_idx': row['video_idx'],
                'error': 'Video file not found'
            })
            print("=" * 80 + "\n")
            continue

        try:
            # 处理样本
            prediction, generated_text, stats = model.process_egoschema_sample(
                video_path=video_path,
                question=row['question'],
                options=row['option']
            )

            # 检查正确性
            gt_answer = int(row['answer'])
            gt_letter = chr(65 + gt_answer)  # 0→A, 1→B, etc.
            correct = (prediction == gt_letter)

            # 更新统计
            total_answered += 1
            if correct:
                total_correct += 1

            # 收集压缩比统计
            if 'stage1_compression_ratio' in stats:
                all_stage1_ratios.append(stats['stage1_compression_ratio'])
            if 'pruning_ratio' in stats:
                all_stage2_ratios.append(stats['pruning_ratio'])
            if 'total_compression_ratio' in stats:
                all_total_ratios.append(stats['total_compression_ratio'])

            # 打印结果
            print(f"\n问题: {row['question'][:100]}...")
            print(f"模型回答: {prediction}")
            print(f"正确答案: {gt_letter}")
            print(f"判断: {'✅ 正确' if correct else '❌ 错误'}")
            print(f"视频采样: {stats.get('num_frames', 0)} 帧")

            # 打印 Token 统计
            print(f"\nToken 统计:")
            if stats.get('tokens_before_stage1', 0) > 0:
                print(f"  Stage 1: {stats['tokens_before_stage1']} → {stats['tokens_after_stage1']} "
                      f"({stats['stage1_compression_ratio']:.1%} 压缩)")
            if stats.get('tokens_before', 0) > 0:
                print(f"  Stage 2: {stats['tokens_before']} → {stats['tokens_after']} "
                      f"({stats['pruning_ratio']:.1%} 压缩)")
            if stats.get('total_compression_ratio', 0) > 0:
                print(f"  总体压缩比: {stats['total_compression_ratio']:.1%}")

            # 打印累计统计
            current_accuracy = total_correct / total_answered if total_answered > 0 else 0.0

            print(f"\n📊 累计统计 (截至第 {sample_idx + 1} 题):")
            print(f"   准确率: {total_correct}/{total_answered} = {current_accuracy:.2%}")

            # Average Token Drop Ratio (最重要的指标) - 即使baseline也显示
            if all_total_ratios:
                avg_token_drop = np.mean(all_total_ratios)
                print(f"   🎯 Average Token Drop Ratio: {avg_token_drop:.2%}")

                # 分阶段统计（可选详细信息）
                if all_stage1_ratios:
                    avg_s1 = np.mean(all_stage1_ratios)
                    print(f"      └─ Stage 1 压缩: {avg_s1:.2%}")
                if all_stage2_ratios:
                    avg_s2 = np.mean(all_stage2_ratios)
                    print(f"      └─ Stage 2 剪枝: {avg_s2:.2%}")
            else:
                # Baseline模式 - 没有pruning
                print(f"   🎯 Average Token Drop Ratio: 0.00% (Baseline - no pruning)")

            # 保存结果
            result = {
                'sample_id': cfg.START_IDX + sample_idx,
                'video_idx': row['video_idx'],
                'question': row['question'],
                'options': row['option'],
                'prediction': prediction,
                'ground_truth': gt_letter,
                'correct': correct,
                'generated_text': generated_text,
                **stats
            }
            results.append(result)

        except Exception as e:
            print(f"\n❌ 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                'sample_id': cfg.START_IDX + sample_idx,
                'video_idx': row['video_idx'],
                'error': str(e)
            })

        print("=" * 80 + "\n")

        # 定期保存
        if (sample_idx + 1) % cfg.SAVE_INTERVAL == 0:
            with open(output_dir / f"results_interim_{sample_idx + 1}.json", 'w') as f:
                json.dump(convert_to_serializable(results), f, indent=2)
            print(f"💾 中间结果已保存\n")

    # 计算最终统计
    print("\n" + "=" * 80)
    print("🎉 评估完成 - 最终结果")
    print("=" * 80 + "\n")

    valid_results = [r for r in results if 'correct' in r]
    num_correct = sum(r['correct'] for r in valid_results)
    num_total = len(valid_results)
    final_accuracy = num_correct / num_total if num_total > 0 else 0.0

    print(f"📈 准确率 (Accuracy):")
    print(f"   {num_correct}/{num_total} = {final_accuracy:.2%}")

    # ============================================================
    # 🎯 Average Token Drop Ratio (核心指标)
    # ============================================================
    if all_total_ratios:
        avg_token_drop = np.mean(all_total_ratios)
        print(f"\n🎯 Average Token Drop Ratio:")
        print(f"   {avg_token_drop:.2%} of tokens were dropped")
        print(f"   {1 - avg_token_drop:.2%} of tokens were kept")
    else:
        print(f"\n🎯 Average Token Drop Ratio:")
        print(f"   0.00% (Baseline - no pruning applied)")

    # 分阶段详细统计
    if cfg.ENABLE_STAGE1 and all_stage1_ratios:
        avg_s1 = np.mean(all_stage1_ratios)
        print(f"\n✂️  Stage 1 详情 (Spatial-Temporal Merging):")
        print(f"   平均压缩: {avg_s1:.2%}")
        print(f"   保留: {1 - avg_s1:.2%}")

    if cfg.ENABLE_PRUNING and all_stage2_ratios:
        avg_s2 = np.mean(all_stage2_ratios)
        print(f"\n✂️  Stage 2 详情 (Attention-Based Pruning):")
        print(f"   平均剪枝: {avg_s2:.2%}")
        print(f"   保留: {1 - avg_s2:.2%}")

    # 时间统计
    inference_times = [r['inference_time'] for r in results if 'inference_time' in r]
    if inference_times:
        avg_time = np.mean(inference_times)
        print(f"\n⏱️  性能:")
        print(f"   平均推理时间: {avg_time:.2f}s/样本")

    # 保存最终结果
    with open(output_dir / "results.json", 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n💾 完整结果已保存: {output_dir / 'results.json'}")

    # 保存摘要
    summary = {
        'experiment': cfg.EXP_NAME,
        'config': config_dict,
        'results': {
            'num_samples': num_total,
            'num_correct': num_correct,
            'accuracy': float(final_accuracy),
            'avg_stage1_compression': float(np.mean(all_stage1_ratios)) if all_stage1_ratios else None,
            'avg_stage2_pruning': float(np.mean(all_stage2_ratios)) if all_stage2_ratios else None,
            'avg_total_compression': float(np.mean(all_total_ratios)) if all_total_ratios else None,
            'avg_inference_time': float(avg_time) if inference_times else None,
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2, ensure_ascii=False)
    print(f"💾 摘要已保存: {output_dir / 'summary.json'}")

    print("\n" + "=" * 80)
    print(f"✅ 实验 '{cfg.EXP_NAME}' 完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
