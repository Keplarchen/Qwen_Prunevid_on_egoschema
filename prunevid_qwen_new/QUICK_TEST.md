# 快速测试指南

## 测试只启用Stage 2（推荐）

在 `eval_egoschema.py` 中修改配置：

```python
# 将 CONFIG_MODE 改为 "custom"
CONFIG_MODE = "custom"

# 自定义配置
CUSTOM_ENABLE_STAGE1 = False      # ⚠️ 禁用Stage 1（避免索引越界）
CUSTOM_ENABLE_STAGE2 = True       # ✅ 启用Stage 2
CUSTOM_KEEP_RATIO = 0.5           # 保留50%的视觉tokens
CUSTOM_PRUNING_LAYER = 10
CUSTOM_ENABLE_CACHE_COMPRESSION = False  # 禁用Stage 3

# 其他设置
NUM_SAMPLES = 10  # 先测试10个样本
VERBOSE = True
```

然后运行：
```bash
cd /mnt/ssd_ext/huggingface/prunevid_qwen_new
python eval_egoschema.py
```

## 预期输出

在每个问题的处理过程中，你应该看到：
```
[Stage 2] 交叉注意力: 12 问题tokens → 128 视觉tokens
[Stage 2] 选择 64/128 视觉tokens (50.0%)
```

**如果看到这个输出，说明修复成功！**

## 预期Accuracy

- **修复前**: ~20% (随机猜测)
- **修复后**: 40-60% (取决于keep_ratio和数据集难度)

## 测试不同的keep_ratio

尝试不同的保留比例，找到最佳平衡：

| keep_ratio | 保留tokens | 预期效果 |
|-----------|----------|---------|
| 0.3 | 30% | 更高压缩，可能accuracy稍低 |
| 0.5 | 50% | 平衡 |
| 0.7 | 70% | 更多信息，accuracy更高 |

## 如果仍然有问题

1. 检查是否看到Stage 2的输出
2. 检查问题tokens数量是否 > 0
3. 分享console输出，特别是Stage 2的信息
