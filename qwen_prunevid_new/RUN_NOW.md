# 🚀 立即运行指南

## ✅ 代码已就绪！

所有代码检查都已通过：
- ✅ PruneVid Stage 1 函数已添加
- ✅ 参数正确传递
- ✅ Stage 1 在正确位置调用（position_embeddings 之后）
- ✅ model_wrapper 已集成

## 🎯 现在就可以运行！

### 方式 1: 运行 EgoSchema 评估（推荐）

```bash
cd /mnt/ssd_ext/huggingface/qwen_prunevid_new
python eval_egoschema.py
```

**配置参数**（在 `eval_egoschema.py` 第 33-37 行）：
```python
ENABLE_STAGE1 = True        # 启用 Stage 1
TAU = 0.8                   # 静态/动态阈值
CLUSTER_RATIO = 0.5         # 聚类比例
NUM_SAMPLES = 10            # 测试样本数（改为 None 可测试全部）
```

### 方式 2: 测试单个视频

1. 编辑 `run_simple_test.py`，修改视频路径：
   ```python
   VIDEO_PATH = "/your/path/to/video.mp4"
   ```

2. 运行：
   ```bash
   python run_simple_test.py
   ```

## 📊 预期输出

如果 `VERBOSE = True`，你会看到：

```
[Stage 1] Tokens: 2304 -> 1152 (50.0% reduction)
```

这表示 PruneVid Stage 1 成功压缩了视频 tokens！

## 🔧 参数调优

### 更激进的压缩（更快，可能略降精度）
```python
TAU = 0.85                     # 更多静态 token
CLUSTER_RATIO = 0.4            # 更激进的聚类
TEMPORAL_SEGMENT_RATIO = 0.2   # 更少的时序段
```

### 更保守的压缩（更慢，更高精度）
```python
TAU = 0.75                     # 更少静态 token
CLUSTER_RATIO = 0.6            # 更保守的聚类
TEMPORAL_SEGMENT_RATIO = 0.3   # 更多的时序段
```

## ❓ 如果遇到问题

### ImportError 问题
环境中的 torchvision/torch 版本不兼容警告可以忽略，不影响运行。

### 看不到压缩效果
确保：
1. `ENABLE_STAGE1 = True`
2. `VERBOSE = True`
3. 输入是视频而不是图片
4. 视频帧数 > 1

### 显存不足
减少 `MAX_FRAMES` 或 `MAX_PIXELS`

## 🎉 开始运行吧！

```bash
python eval_egoschema.py
```

祝测试顺利！🚀
