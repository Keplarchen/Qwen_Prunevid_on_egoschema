# EgoSchema评估指南

## 快速开始

### 1. 准备数据

确保你有EgoSchema数据集的视频文件：

```bash
# 视频应该在这个目录下
/mnt/ssd_ext/huggingface/egoschema/videos/
```

### 2. 配置评估参数

编辑 `eval_egoschema.py` 文件顶部的配置：

```python
# 选择配置模式
CONFIG_MODE = "paper"  # 可选: "baseline", "paper", "conservative", "aggressive", "custom"

# 测试样本数
NUM_SAMPLES = 10  # None表示全部

# 视频帧数
MAX_FRAMES = 16
```

### 3. 运行评估

```bash
cd /mnt/ssd_ext/huggingface/prunevid_qwen_new

python eval_egoschema.py
```

## 配置模式说明

### 1. baseline
无token剪枝，用于对比

```python
CONFIG_MODE = "baseline"
```

### 2. paper (推荐)
论文推荐配置：
- tau = 0.8
- cluster_ratio = 0.5
- keep_ratio = 0.4
- 预期token保留率: 15-20%

```python
CONFIG_MODE = "paper"
```

### 3. conservative
高压缩配置，追求更高效率：
- tau = 0.7
- cluster_ratio = 0.3
- keep_ratio = 0.3
- 更快但准确率可能略降

```python
CONFIG_MODE = "conservative"
```

### 4. aggressive
低压缩配置，追求更高准确率：
- tau = 0.85
- cluster_ratio = 0.6
- keep_ratio = 0.5
- 更准确但速度适中

```python
CONFIG_MODE = "aggressive"
```

### 5. custom
自定义配置：

```python
CONFIG_MODE = "custom"

# 然后修改CUSTOM_开头的参数
CUSTOM_ENABLE_STAGE1 = True
CUSTOM_TAU = 0.8
CUSTOM_CLUSTER_RATIO = 0.5
# ...
```

## 输出说明

### 运行时输出

每个样本会显示：
- 问题和答案
- 准确率统计
- Token压缩统计

```
Sample 1/10
Video ID: xxx
Question: ...
Ground Truth: 2
Predicted:    2
Correct: ✓

📊 Current Accuracy: 1/1 = 100.00%

📉 Token Compression (Current Sample):
  Original:      1024
  After Stage 1: 512 (drop: 50.0%)
  After Stage 2: 205 (drop: 60.0%)
  Total drop:    80.0%
```

### 最终结果

```
🎉 EVALUATION COMPLETED

📊 Final Accuracy:
  Correct: 8/10
  Accuracy: 80.00%

📉 Final Token Compression:
  Total tokens before:       10240
  Total tokens after Stage 1: 5120
  Total tokens after Stage 2: 2048

  Stage 1 drop ratio: 50.00%
  Stage 2 drop ratio: 60.00%
  Total drop ratio:   80.00%

⏱️  Time Statistics:
  Total time: 120.50s
  Avg time per sample: 12.05s
```

### 保存的结果文件

结果会保存在 `./results/` 目录：

```
results/egoschema_results_20251102_174530_paper.json
```

JSON格式：
```json
{
  "config": {
    "config_mode": "paper",
    "model_config": {...},
    ...
  },
  "summary": {
    "total_samples": 10,
    "correct_samples": 8,
    "accuracy": 80.0,
    "stage1_drop_ratio": 50.0,
    "stage2_drop_ratio": 60.0,
    "total_drop_ratio": 80.0
  },
  "results": [...]
}
```

## 常见用法

### 快速测试（10个样本）

```python
NUM_SAMPLES = 10
CONFIG_MODE = "paper"
```

### 完整评估（所有样本）

```python
NUM_SAMPLES = None  # 全部
CONFIG_MODE = "paper"
```

### 只测试Stage 1

```python
CONFIG_MODE = "custom"
CUSTOM_ENABLE_STAGE1 = True
CUSTOM_ENABLE_STAGE2 = False
CUSTOM_ENABLE_CACHE_COMPRESSION = False
```

### 调整视频帧数

```python
MAX_FRAMES = 32  # 增加到32帧
```

## 性能优化建议

### 1. 减少帧数加快测试
```python
MAX_FRAMES = 8  # 更快
```

### 2. 使用更小的样本集
```python
NUM_SAMPLES = 5
```

### 3. 关闭详细输出
```python
VERBOSE = False
```

## 故障排查

### 问题1: 视频文件未找到

**现象**: `Warning: Video not found: ...`

**解决**:
- 检查 `VIDEO_DIR` 路径是否正确
- 确认视频文件存在

### 问题2: CUDA Out of Memory

**解决**:
```python
MAX_FRAMES = 8  # 减少帧数
CONFIG_MODE = "conservative"  # 使用高压缩配置
```

### 问题3: 模型加载失败

**解决**:
- 检查 `MODEL_PATH` 是否正确
- 确保有网络连接（如果从HF下载）
- 或使用本地路径：
  ```python
  MODEL_PATH = "/path/to/local/qwen2.5-vl-7b"
  ```

## 对比不同配置

运行脚本多次，使用不同的CONFIG_MODE：

```bash
# Baseline
sed -i 's/CONFIG_MODE = .*/CONFIG_MODE = "baseline"/' eval_egoschema.py
python eval_egoschema.py

# Paper
sed -i 's/CONFIG_MODE = .*/CONFIG_MODE = "paper"/' eval_egoschema.py
python eval_egoschema.py

# Conservative
sed -i 's/CONFIG_MODE = .*/CONFIG_MODE = "conservative"/' eval_egoschema.py
python eval_egoschema.py
```

然后对比 `./results/` 下的结果文件。

## 高级用法

### 分段评估

评估前100个样本：
```python
START_INDEX = 0
NUM_SAMPLES = 100
```

继续评估下100个：
```python
START_INDEX = 100
NUM_SAMPLES = 100
```

### 只评估特定范围

例如样本50-60：
```python
START_INDEX = 50
NUM_SAMPLES = 10
```

## 预期性能

基于论文结果，在Qwen2.5-VL上预期：

| 配置 | 准确率 | Token保留率 | 加速比 |
|------|--------|-------------|--------|
| Baseline | ~60% | 100% | 1.0× |
| Paper | ~60% | 15-20% | 1.5-2.0× |
| Conservative | ~58% | 10-15% | 2.0-2.5× |
| Aggressive | ~61% | 25-30% | 1.3-1.5× |

**注**: 实际性能会因硬件和具体实现而异。

## 联系支持

如遇问题：
1. 查看 README.md
2. 检查代码注释
3. 提交Issue
