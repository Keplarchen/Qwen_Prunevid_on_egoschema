# 🔧 显存溢出 (OOM) 快速修复指南

## 问题

遇到显存溢出错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 377.29 GiB
```

## 原因

Stage 1 改变序列长度后，attention mask 计算出现问题，导致 attention 矩阵尺寸异常。

## ✅ 已应用的修复

1. ✅ 添加了 causal_mask 重新计算逻辑
2. ✅ 添加了错误处理和安全检查
3. ✅ 在 Stage 1 出错时自动跳过

## 🚀 立即解决方案

### 方案 1: 先禁用 Stage 1 测试基础功能（推荐）

编辑 `eval_egoschema.py` 第 33 行：

```python
# 将 ENABLE_STAGE1 改为 False
ENABLE_STAGE1 = False  # 暂时禁用
```

然后运行：
```bash
python eval_egoschema.py
```

这样可以先确保基础模型正常工作，然后再逐步调试 Stage 1。

### 方案 2: 降低视频分辨率和帧数

如果想测试 Stage 1，可以先降低资源消耗：

编辑 `eval_egoschema.py`:

```python
# 第 47-51 行，降低这些值
MAX_FRAMES = 8  # 从 16 降到 8
MAX_PIXELS = 192 * 192 * 8  # 对应减少

# 第 33 行，启用 Stage 1
ENABLE_STAGE1 = True

# 第 59 行，减少测试样本
NUM_SAMPLES = 1  # 先测试 1 个样本
```

### 方案 3: 使用更保守的 Stage 1 参数

```python
ENABLE_STAGE1 = True
TAU = 0.7  # 降低阈值，减少静态 token
CLUSTER_RATIO = 0.7  # 提高比例，减少聚类压缩
TEMPORAL_SEGMENT_RATIO = 0.4  # 增加分段，减少每段大小
```

## 🔍 调试步骤

### 1. 测试基础模型（不使用 Stage 1）

```bash
# 确保 ENABLE_STAGE1 = False
python eval_egoschema.py
```

**预期输出**: 能正常运行并生成结果

### 2. 测试单个样本 + Stage 1

```python
ENABLE_STAGE1 = True
NUM_SAMPLES = 1  # 只测试 1 个
MAX_FRAMES = 8   # 降低帧数
VERBOSE = True   # 看详细信息
```

**预期输出**:
- 如果成功，会看到 `[Stage 1] Tokens: XXX -> YYY (ZZ% reduction)`
- 如果失败，会看到 `[Stage 1] ... skipping` 并继续运行

### 3. 逐步增加负载

一旦单个样本成功：
```python
NUM_SAMPLES = 5   # 增加到 5 个
MAX_FRAMES = 12   # 逐步增加帧数
```

## 📊 内存使用建议

### GPU 显存: 48GB (你的配置)

**保守配置**（确保不 OOM）：
```python
MAX_FRAMES = 8
MAX_PIXELS = 192 * 192 * 8  # ~295K pixels
ENABLE_STAGE1 = True
```

**标准配置**（应该可以）：
```python
MAX_FRAMES = 12
MAX_PIXELS = 192 * 192 * 12  # ~442K pixels
ENABLE_STAGE1 = True
```

**激进配置**（可能会紧张）：
```python
MAX_FRAMES = 16
MAX_PIXELS = 192 * 192 * 16  # ~589K pixels
ENABLE_STAGE1 = True
```

## ⚠️ 如果仍然 OOM

1. **检查其他进程**:
   ```bash
   nvidia-smi  # 查看 GPU 使用情况
   ```

2. **清理 GPU 缓存**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **降低批处理大小**:
   eval_egoschema.py 默认 batch_size=1，已经是最小值

4. **使用 CPU offloading**:
   ```python
   # 在 model_wrapper.py 中修改
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
       ...,
       device_map="auto",  # 改为 auto，自动分配
   )
   ```

## 🎯 推荐测试流程

```bash
# 1. 先测试基础模型
# 设置: ENABLE_STAGE1 = False, NUM_SAMPLES = 5
python eval_egoschema.py

# 2. 测试 Stage 1（单样本）
# 设置: ENABLE_STAGE1 = True, NUM_SAMPLES = 1, VERBOSE = True
python eval_egoschema.py

# 3. 如果成功，增加样本数
# 设置: NUM_SAMPLES = 10
python eval_egoschema.py

# 4. 全面测试
# 设置: NUM_SAMPLES = None (所有样本)
python eval_egoschema.py
```

## 📝 注意事项

- Stage 1 的 OOM 问题已经通过重新计算 causal_mask 修复
- 但为了安全，建议先禁用 Stage 1 测试基础功能
- Stage 1 在遇到错误时会自动跳过，不会导致程序崩溃
- 所有修改都已保存在 modeling_qwen2_5_vl_prunevid_full.py 中

## ✅ 当前状态

- ✅ 代码已修复
- ✅ 错误处理已添加
- ⚠️ 建议先禁用 Stage 1 测试
- 🔄 Stage 1 需要进一步测试和优化

## 联系

如果问题持续，请检查：
1. GPU 型号和显存大小
2. PyTorch 和 CUDA 版本
3. 其他占用 GPU 的进程
