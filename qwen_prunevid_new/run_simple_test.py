"""
简单测试脚本 - 使用新的 PruneVid Stage 1 集成模型

这个脚本展示如何使用集成了 PruneVid Stage 1 的 Qwen2.5-VL 模型
"""

import torch
import sys
from pathlib import Path

# 添加当前目录到 Python path
sys.path.insert(0, str(Path(__file__).parent))

# 导入新的 modeling 文件
from modeling_qwen2_5_vl_prunevid_full import Qwen2_5_VLForConditionalGeneration

# 兼容不同版本的 transformers
try:
    from transformers import AutoProcessor
except ImportError:
    try:
        from transformers import Qwen2VLProcessor as AutoProcessor
    except ImportError:
        print("警告: 无法导入 AutoProcessor，将使用 AutoTokenizer")
        from transformers import AutoTokenizer as AutoProcessor

from qwen_vl_utils import process_vision_info

# ============================================================================
# 配置参数
# ============================================================================

# 模型路径
MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"

# 视频路径（修改为你的视频路径）
VIDEO_PATH = "/mnt/ssd_ext/huggingface/egoschema/videos/test_video.mp4"

# 问题
QUESTION = "What is happening in this video?"

# PruneVid Stage 1 参数
ENABLE_STAGE1 = True  # 是否启用 Stage 1
TAU = 0.8  # 静态/动态分离阈值
CLUSTER_RATIO = 0.5  # 空间聚类保留比例
TEMPORAL_SEGMENT_RATIO = 0.25  # 时序分段比例
DPC_KNN_K = 5  # DPC-KNN 的 k 近邻参数
VERBOSE = True  # 是否打印详细信息

# 视频处理参数
MAX_FRAMES = 16  # 最大帧数
MIN_PIXELS = 224 * 224
MAX_PIXELS = 192 * 192 * MAX_FRAMES  # 589824 < 602112

# 生成参数
MAX_NEW_TOKENS = 100

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 80)
    print("PruneVid Stage 1 + Qwen2.5-VL 测试")
    print("=" * 80)

    # 1. 加载模型和处理器
    print("\n[1/5] 加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"

    print(f"✅ 模型加载完成: {MODEL_PATH}")

    # 2. 准备输入
    print("\n[2/5] 准备视频输入...")

    # 检查视频文件是否存在
    if not Path(VIDEO_PATH).exists():
        print(f"❌ 错误: 视频文件不存在: {VIDEO_PATH}")
        print("请修改 VIDEO_PATH 变量为有效的视频路径")
        return

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": VIDEO_PATH,
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                    "fps": 1.0,
                },
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 处理视频
    image_inputs, video_inputs = process_vision_info(messages)

    # 准备模型输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

    print(f"✅ 视频加载完成: {VIDEO_PATH}")
    print(f"   输入序列长度: {inputs['input_ids'].shape[1]}")

    # 3. 不使用 Stage 1 的推理（baseline）
    if ENABLE_STAGE1:
        print("\n[3/5] Baseline 推理（不使用 Stage 1）...")
        with torch.no_grad():
            baseline_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                enable_stage1=False,  # 禁用 Stage 1
            )

        baseline_text = processor.batch_decode(
            baseline_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"✅ Baseline 生成完成")
        print(f"   回答: {baseline_text[-200:]}")  # 只显示最后200字符

    # 4. 使用 Stage 1 的推理
    print(f"\n[4/5] 使用 PruneVid Stage 1 推理...")
    print(f"   参数: tau={TAU}, cluster_ratio={CLUSTER_RATIO}")
    print(f"         temporal_segment_ratio={TEMPORAL_SEGMENT_RATIO}, k={DPC_KNN_K}")

    with torch.no_grad():
        prunevid_outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # PruneVid Stage 1 参数
            enable_stage1=ENABLE_STAGE1,
            tau=TAU,
            cluster_ratio=CLUSTER_RATIO,
            temporal_segment_ratio=TEMPORAL_SEGMENT_RATIO,
            dpc_knn_k=DPC_KNN_K,
            verbose=VERBOSE,
        )

    prunevid_text = processor.batch_decode(
        prunevid_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"✅ PruneVid 生成完成")

    # 5. 显示结果
    print("\n" + "=" * 80)
    print("[5/5] 结果对比")
    print("=" * 80)

    if ENABLE_STAGE1:
        print("\n【Baseline 回答】")
        print(baseline_text)
        print("\n" + "-" * 80)

    print("\n【PruneVid Stage 1 回答】")
    print(prunevid_text)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
