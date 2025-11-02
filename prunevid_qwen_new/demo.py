"""
PruneVid演示脚本
==============

这个脚本展示如何使用PruneVid进行视频理解。

使用方法：
python demo.py --video_path <视频路径> --question "你的问题"

或者使用默认示例：
python demo.py
"""

import argparse
from pathlib import Path
import sys

# 添加当前目录到path以便导入
sys.path.insert(0, str(Path(__file__).parent))

from prunevid_qwen_new import PruneVidQwen25VL, get_paper_config, get_baseline_config


def main():
    parser = argparse.ArgumentParser(description="PruneVid视频理解演示")

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL模型路径或HuggingFace model ID"
    )

    # 输入参数
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="视频文件路径"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="请详细描述视频中发生的事情。",
        help="要问的问题"
    )

    # PruneVid配置
    parser.add_argument(
        "--config",
        type=str,
        default="paper",
        choices=["baseline", "paper", "conservative", "aggressive"],
        help="PruneVid配置: baseline(无剪枝), paper(论文推荐), conservative(高压缩), aggressive(低压缩)"
    )

    # 生成参数
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备"
    )

    # 调试参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志"
    )

    args = parser.parse_args()

    # 检查视频路径
    if args.video_path is None:
        print("错误：必须提供--video_path参数")
        print("\n使用示例：")
        print("  python demo.py --video_path /path/to/video.mp4 --question \"视频中发生了什么？\"")
        return

    if not Path(args.video_path).exists():
        print(f"错误：视频文件不存在: {args.video_path}")
        return

    # 选择配置
    print(f"\n使用配置: {args.config}")
    if args.config == "baseline":
        from prunevid_qwen_new import get_baseline_config
        config = get_baseline_config()
    elif args.config == "paper":
        from prunevid_qwen_new import get_paper_config
        config = get_paper_config()
    elif args.config == "conservative":
        from prunevid_qwen_new import get_conservative_config
        config = get_conservative_config()
    elif args.config == "aggressive":
        from prunevid_qwen_new import get_aggressive_config
        config = get_aggressive_config()

    # 设置verbose
    if args.verbose:
        config.verbose = True

    # 加载模型
    print("\n" + "=" * 60)
    print("PruneVid for Qwen2.5-VL - 视频理解演示")
    print("=" * 60)

    model = PruneVidQwen25VL(
        model_path=args.model_path,
        config=config,
        device=args.device,
    )

    # 生成回答
    print(f"\n视频: {args.video_path}")
    print(f"问题: {args.question}\n")
    print("正在处理...")

    result = model.generate(
        video_path=args.video_path,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        return_stats=True,
    )

    # 显示结果
    print("\n" + "=" * 60)
    print("生成结果")
    print("=" * 60)
    print(f"\n回答：\n{result['answer']}\n")
    print(f"输入tokens: {result['input_tokens']}")
    print(f"输出tokens: {result['output_tokens']}")

    # 显示统计信息（已在generate中打印）

    print("\n演示完成！")


def demo_comparison():
    """
    对比演示：baseline vs PruneVid

    展示PruneVid的加速效果
    """
    print("\n" + "=" * 60)
    print("PruneVid对比演示")
    print("=" * 60)

    import time

    # 需要提供一个示例视频
    video_path = "/path/to/sample_video.mp4"  # 请替换为实际路径
    question = "视频中的主要内容是什么？"

    if not Path(video_path).exists():
        print(f"请将示例视频放在: {video_path}")
        return

    # 1. Baseline（无剪枝）
    print("\n[1/2] 运行Baseline（无剪枝）...")
    config_baseline = get_baseline_config()
    model_baseline = PruneVidQwen25VL(
        config=config_baseline,
        device="cuda",
    )

    start = time.time()
    result_baseline = model_baseline.generate(
        video_path=video_path,
        question=question,
        return_stats=False,
    )
    time_baseline = time.time() - start

    # 2. PruneVid（论文配置）
    print("\n[2/2] 运行PruneVid（论文配置）...")
    config_prunevid = get_paper_config()
    model_prunevid = PruneVidQwen25VL(
        config=config_prunevid,
        device="cuda",
    )

    start = time.time()
    result_prunevid = model_prunevid.generate(
        video_path=video_path,
        question=question,
        return_stats=True,
    )
    time_prunevid = time.time() - start

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    print(f"\n方法           | 时间(s) | 加速比")
    print("-" * 40)
    print(f"Baseline       | {time_baseline:.2f}   | 1.00x")
    print(f"PruneVid       | {time_prunevid:.2f}   | {time_baseline/time_prunevid:.2f}x")

    print(f"\n回答（Baseline）:")
    print(result_baseline['answer'])

    print(f"\n回答（PruneVid）:")
    print(result_prunevid['answer'])

    print("\n对比完成！")


if __name__ == "__main__":
    # 运行主演示
    main()

    # 如果需要运行对比演示，取消下面的注释
    # demo_comparison()
