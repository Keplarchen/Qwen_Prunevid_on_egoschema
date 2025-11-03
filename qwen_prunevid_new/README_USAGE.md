# 如何运行 PruneVid Stage 1 + Qwen2.5-VL

本文档说明如何使用集成了 PruneVid Stage 1 的 Qwen2.5-VL 模型。

## 目录结构

```
qwen_prunevid_new/
├── modeling_qwen2_5_vl_prunevid_full.py  # 核心：集成了 PruneVid Stage 1 的完整模型
├── model_wrapper.py                      # 已修改：使用新的 modeling 文件
├── config.py                             # PruneVid 配置
├── eval_egoschema.py                     # EgoSchema 评估脚本
├── run_simple_test.py                    # 简单测试脚本（新创建）
└── README_USAGE.md                       # 本文件
```

## 快速开始

### 方法 1: 使用简单测试脚本（推荐）

这是最简单的方式，用于快速测试 PruneVid Stage 1 的效果。

1. **修改配置**

   编辑 `run_simple_test.py`，修改以下参数：

   ```python
   # 模型路径
   MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"

   # 视频路径（修改为你的视频路径）
   VIDEO_PATH = "/path/to/your/video.mp4"

   # 问题
   QUESTION = "What is happening in this video?"

   # PruneVid Stage 1 参数
   ENABLE_STAGE1 = True
   TAU = 0.8  # 静态/动态分离阈值 (0.6-0.9)
   CLUSTER_RATIO = 0.5  # 空间聚类保留比例 (0.3-0.7)
   TEMPORAL_SEGMENT_RATIO = 0.25  # 时序分段比例 (0.125-0.5)
   DPC_KNN_K = 5  # DPC-KNN 的 k 近邻参数
   VERBOSE = True  # 是否打印详细信息
   ```

2. **运行测试**

   ```bash
   cd /mnt/ssd_ext/huggingface/qwen_prunevid_new
   python run_simple_test.py
   ```

3. **查看输出**

   脚本会输出：
   - Baseline（不使用 Stage 1）的回答
   - PruneVid Stage 1 的回答
   - Token 压缩统计信息（如果 verbose=True）

### 方法 2: 使用 Wrapper 类

这种方式更灵活，适合集成到自己的代码中。

```python
import sys
sys.path.insert(0, '/mnt/ssd_ext/huggingface/qwen_prunevid_new')

from config import PruneVidConfig
from model_wrapper import Qwen25VLPruneVid

# 1. 配置 PruneVid
config = PruneVidConfig(
    # Stage 1 参数
    enable_stage1=True,
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
    dpc_knn_k=5,

    # Stage 2 参数（可选）
    enable_pruning=False,  # 暂时禁用 Stage 2

    # 其他配置
    verbose=True,
)

# 2. 加载模型
model = Qwen25VLPruneVid(
    model_path="/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct",
    config=config,
    device="cuda:0",
)

# 3. 生成回答
result = model.generate(
    video_path="/path/to/your/video.mp4",
    question="What is happening in this video?",
    max_new_tokens=100,
    return_dict=True,
)

# 4. 查看结果
print("回答:", result['generated_text'])
print("压缩统计:", result['compression_stats'])
print("生成时间:", result['generation_time'], "秒")
```

### 方法 3: 直接使用模型（最灵活）

如果你想完全控制推理过程：

```python
import torch
import sys
sys.path.insert(0, '/mnt/ssd_ext/huggingface/qwen_prunevid_new')

from modeling_qwen2_5_vl_prunevid_full import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

processor = AutoProcessor.from_pretrained(
    "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"
)

# 2. 准备输入
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/path/to/your/video.mp4",
                "max_pixels": 589824,  # 192*192*16
                "fps": 1.0,
            },
            {"type": "text", "text": "What is happening in this video?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to("cuda:0")

# 3. 生成（带 PruneVid Stage 1）
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        # PruneVid Stage 1 参数
        enable_stage1=True,
        tau=0.8,
        cluster_ratio=0.5,
        temporal_segment_ratio=0.25,
        dpc_knn_k=5,
        verbose=True,  # 打印压缩信息
    )

# 4. 解码
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(result)
```

### 方法 4: 运行 EgoSchema 评估

如果你想在 EgoSchema 数据集上评估：

1. **修改配置**

   编辑 `eval_egoschema.py` 的配置区域（第 16-63 行）：

   ```python
   # 数据集配置
   VIDEO_DIR = "/mnt/ssd_ext/huggingface/egoschema/videos"

   # 模型配置
   MODEL_PATH = "/mnt/ssd_ext/huggingface/models/Qwen2.5-VL-7B-Instruct"

   # Stage 1 参数
   ENABLE_STAGE1 = True
   TAU = 0.8
   CLUSTER_RATIO = 0.5
   TEMPORAL_SEGMENT_RATIO = 0.25

   # 测试配置
   NUM_SAMPLES = 10  # 测试样本数量，None 表示全部
   ```

2. **运行评估**

   ```bash
   cd /mnt/ssd_ext/huggingface/qwen_prunevid_new
   python eval_egoschema.py
   ```

## PruneVid Stage 1 参数说明

### 核心参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `enable_stage1` | `False` | `True/False` | 是否启用 Stage 1 |
| `tau` | `0.8` | `0.6-0.9` | 静态/动态分离阈值，越大越多 token 被视为静态 |
| `cluster_ratio` | `0.5` | `0.3-0.7` | 空间聚类保留比例，越小压缩越多 |
| `temporal_segment_ratio` | `0.25` | `0.125-0.5` | 时序分段比例，越小分段越少 |
| `dpc_knn_k` | `5` | `3-10` | DPC-KNN 的 k 近邻参数 |
| `verbose` | `False` | `True/False` | 是否打印详细的压缩信息 |

### 参数调优建议

**压缩更多 tokens (更快，可能略降低精度):**
```python
tau=0.85                     # 更多静态 token
cluster_ratio=0.4            # 更激进的聚类
temporal_segment_ratio=0.2   # 更少的时序段
```

**更保守的压缩 (更慢，更高精度):**
```python
tau=0.75                     # 更少静态 token
cluster_ratio=0.6            # 更保守的聚类
temporal_segment_ratio=0.3   # 更多的时序段
```

**论文推荐配置 (平衡):**
```python
tau=0.8
cluster_ratio=0.5
temporal_segment_ratio=0.25
dpc_knn_k=5
```

## 工作原理

PruneVid Stage 1 在 **position_embeddings 生成之后、decoder layers 之前** 执行 token 压缩：

```
输入视频
  ↓
视觉编码器 (Vision Encoder)
  ↓
嵌入层 (Embed Tokens)
  ↓
位置编码生成 (Position Embeddings)
  ↓
🎯 PruneVid Stage 1: 时空 Token 合并
  │  1. 时序聚类：将帧分组为场景段
  │  2. 静态/动态分离：识别静态区域
  │  3. 静态 token 合并：时间维度平均
  │  4. 动态 token 聚类：空间聚类
  ↓
解码器层 (Decoder Layers)
  ↓
生成输出
```

## 预期效果

根据 PruneVid 论文，在视频任务上：

- **Token 压缩率**: 通常可以减少 40-60% 的 visual tokens
- **性能影响**: 在大多数基准测试上保持或略微提升性能
- **速度提升**: 推理速度提升 1.5-2x（取决于压缩率）

## 常见问题

### Q1: 为什么没有看到压缩效果？

A: 确保：
1. `enable_stage1=True`
2. `verbose=True` 以查看压缩信息
3. 输入是**视频**而不是图片
4. 视频帧数 > 1

### Q2: 如何确认 Stage 1 正在工作？

A: 设置 `verbose=True`，你会看到类似输出：
```
[Stage 1] Tokens: 2304 -> 1152 (50.0% reduction)
```

### Q3: 可以同时使用 Stage 1 和 Stage 2 吗？

A: 可以，但 Stage 2 (基于注意力的剪枝) 目前可能需要额外调整。建议先只使用 Stage 1。

### Q4: 压缩后结果变差了怎么办？

A: 尝试：
1. 减小 `tau` (例如 0.75)
2. 增大 `cluster_ratio` (例如 0.6)
3. 增大 `temporal_segment_ratio` (例如 0.3)

## 性能基准

在 EgoSchema 数据集上的测试结果（预期）：

| 配置 | Token 保留率 | 准确率 | 推理速度 |
|------|-------------|--------|---------|
| Baseline (无 Stage 1) | 100% | - | 1x |
| Stage 1 (推荐配置) | ~45% | ~98% | ~1.8x |
| Stage 1 (激进) | ~30% | ~95% | ~2.3x |

## 下一步

1. **调优参数**: 根据你的任务调整 `tau`, `cluster_ratio` 等参数
2. **集成 Stage 2**: 添加基于注意力的 token 选择（如需要）
3. **评估性能**: 在你的数据集上评估压缩率和精度权衡

## 技术支持

如有问题，请检查：
1. 模型路径是否正确
2. 视频文件是否存在
3. GPU 内存是否足够
4. transformers 版本是否兼容（建议 >= 4.37.0）
