# PruneVid快速开始指南

## 5分钟快速上手

### 步骤1: 安装依赖

```bash
pip install transformers>=4.40.0 accelerate torch opencv-python pillow
pip install qwen-vl-utils
```

### 步骤2: 准备视频

准备一个测试视频，例如 `test_video.mp4`

### 步骤3: 运行示例

```python
import sys
sys.path.insert(0, '/mnt/ssd_ext/huggingface')

from prunevid_qwen_new import PruneVidQwen25VL, get_paper_config

# 加载模型
model = PruneVidQwen25VL(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    config=get_paper_config(),
    device="cuda",
)

# 生成回答
result = model.generate(
    video_path="test_video.mp4",
    question="视频中发生了什么？",
)

print(result['answer'])
```

### 步骤4: 查看压缩效果

```python
# 查看详细统计
stats = result['stats']

print(f"原始tokens: {stats['stage1']['original_tokens']}")
print(f"压缩后tokens: {stats['stage1']['compressed_tokens']}")
print(f"压缩率: {stats['stage1']['reduction_percentage']:.1f}%")
```

## 使用命令行

```bash
cd /mnt/ssd_ext/huggingface/prunevid_qwen_new

python demo.py \
    --video_path /path/to/your/video.mp4 \
    --question "描述视频内容" \
    --config paper \
    --verbose
```

## 不同配置对比

```python
from prunevid_qwen_new import (
    get_baseline_config,     # 无剪枝
    get_paper_config,        # 论文配置
    get_conservative_config, # 高压缩
)

# 无剪枝（baseline）
model_baseline = PruneVidQwen25VL(config=get_baseline_config())

# 论文配置（推荐）
model_paper = PruneVidQwen25VL(config=get_paper_config())

# 高压缩（更快）
model_fast = PruneVidQwen25VL(config=get_conservative_config())
```

## 调试模式

```python
from prunevid_qwen_new import PruneVidConfig

config = PruneVidConfig(
    enable_stage1=True,
    enable_stage2=True,
    enable_cache_compression=True,
    verbose=True,  # 打开详细日志
)

model = PruneVidQwen25VL(config=config)
```

## 常见问题

### Q: 如何处理长视频？

A: 调整`max_frames`参数：

```python
config = get_paper_config()
config.max_frames = 32  # 增加采样帧数
model = PruneVidQwen25VL(config=config)
```

### Q: 如何提高准确率？

A: 使用aggressive配置或增加`keep_ratio`:

```python
config = get_aggressive_config()
# 或
config.keep_ratio = 0.5  # 保留50%的tokens
```

### Q: 如何提高速度？

A: 使用conservative配置或降低`keep_ratio`:

```python
config = get_conservative_config()
# 或
config.keep_ratio = 0.3  # 只保留30%
```

### Q: CUDA Out of Memory怎么办？

A: 减少帧数或使用更小的batch size：

```python
config.max_frames = 8  # 减少帧数
```

## 下一步

- 阅读完整文档：[README.md](README.md)
- 查看配置选项：[config.py](config.py)
- 运行demo：`python demo.py --help`
- 在EgoSchema上测试（需要自己编写评估脚本）

## 技术支持

如有问题，请查看：
1. README.md中的详细说明
2. 源代码中的注释
3. 提交Issue到GitHub

祝使用愉快！
