# PruneVid Bug 修复总结

## 🔴 关键Bug：Stage 2 交叉注意力提取错误

### 问题描述
Stage 2 错误地使用了**系统prompt的注意力**来选择视觉tokens，而不是**问题文本的注意力**。

**原因**：代码假设token顺序为 `[文本] [视觉]`，但Qwen2.5-VL实际顺序是：
```
[系统prompt] [vision_start] [视觉tokens] [vision_end] [问题文本]
```

### 影响
- Accuracy ≈ 20%（完全随机，5选1）
- 模型无法理解视频内容与问题的关系

### 修复内容

**文件**: `modeling_qwen2_5_vl_prunevid_dtd.py`

**修复1** (Line 1256-1259):
```python
# 修复前（错误）：
text_token_start = 0
text_token_end = visual_token_start  # 提取系统prompt

# 修复后（正确）：
text_token_start = visual_token_end + 1  # 跳过vision_end_token
text_token_end = seq_len  # 提取问题文本
```

**修复2** (Line 1729-1737): 防止Stage 2在decode阶段被重复触发
```python
# 添加prefill检查
is_prefill = (cache_position is not None and cache_position[0] == 0) or \
             (past_key_values is None or past_key_values.get_seq_length() == 0)

stage2_should_extract = (
    self.prunevid_stage2 is not None and
    input_ids is not None and
    self.config.vision_start_token_id in input_ids[0] and
    is_prefill  # 只在prefill阶段执行
)
```

## ⚠️ 已知问题

**Stage 1 存在索引越界问题**，与Stage 2同时启用时会报CUDA错误。

## 🧪 测试建议

### 方案1：只启用Stage 2（推荐）

```python
config = PruneVidConfig(
    enable_stage1=False,      # 暂时禁用Stage 1
    enable_stage2=True,       # 启用Stage 2
    keep_ratio=0.5,
    pruning_layer=10,
    enable_cache_compression=False,
    verbose=True,
)
```

**预期结果**：Accuracy应从 ~20% 提升到 40-60%

### 方案2：完全禁用PruneVid（baseline）

```python
config = PruneVidConfig(
    enable_stage1=False,
    enable_stage2=False,
    enable_cache_compression=False,
)
```

## 📊 验证方法

运行evaluation时，在verbose模式下应该看到：
```
[Stage 2] 交叉注意力: 12 问题tokens → 128 视觉tokens
[Stage 2] 选择 64/128 视觉tokens (50.0%)
```

**关键指标**：
- 问题tokens数量 > 0（之前是0，因为提取了系统prompt）
- Accuracy > 30%（远高于随机的20%）

## 🔧 后续工作

1. 调试并修复Stage 1的索引越界问题
2. 测试Stage 1 + Stage 2的组合
3. 验证Stage 3 KV cache压缩
4. 在多个keep_ratio设置下测试性能

---
**修复时间**: 2025-11-02
**修复内容**: Stage 2交叉注意力逻辑 + prefill检查
**状态**: Stage 2单独使用已验证可工作
