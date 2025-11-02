# PruneVid Qwen2.5-VL 完整实现说明

## 版本更新

**v2.0** - 完整实现PruneVid三阶段方法：
- ✅ **Stage 1**: Spatial-Temporal Token Merging（Vision Encoder层面）
- ✅ **Stage 2**: Attention-Based Token Pruning（LLM层面）
- ✅ **Stage 3**: KV Cache Compression（LLM层面）

**v1.0** - 仅实现Stage 2和Stage 3

## 实现概述

PruneVid是一个三阶段的视觉token剪枝方法：

### Stage 1: Spatial-Temporal Token Merging（新增）

在vision encoder输出后、进入LLM前进行：

1. **Temporal Segmentation**: 使用DPC-KNN算法将视频帧聚类成temporal segments
2. **Static/Dynamic Separation**: 基于帧间相似度区分静态和动态区域
   - 静态区域：跨时间平均，然后空间聚类
   - 动态区域：每帧独立保留，每帧内空间聚类
3. **Spatial Clustering**: 使用DPC-KNN对spatial tokens进行聚类压缩

**参数**：
- `tau` (0.7-0.9): 静态/动态阈值，相似度>tau为静态
- `cluster_ratio` (0.3-0.8): 空间聚类后保留的token比例
- `temporal_segment_ratio` (0.25-0.5): 时序分段比例

### Stage 2: Attention-Based Token Pruning（已有）

在LLM的第N层（默认第10层）：

1. 提取text-to-visual attention权重
2. 计算visual token重要性：`max(text_dim) → max(heads_dim)`
3. 选择top-k重要的visual tokens
4. 配置custom cache进行pruning

**参数**：
- `keep_ratio` (0.3-0.6): 保留的visual token比例
- `pruning_layer` (8-12): 在哪一层进行pruning

### Stage 3: KV Cache Compression（已有）

在pruning_layer完成时自动触发：

1. 根据Stage 2计算的kept_indices
2. 压缩所有层（0到pruning_layer）的KV cache
3. 真正删除不重要的tokens（seq_len维度减小）

---

## v1.0 修复的Bug

### 🔴 原实现的问题

1. **最严重**：只是mask置零，没有真正删除token
   - KV cache仍存储所有token（无内存节省）
   - Attention仍计算所有token（无计算加速）
2. Attention重要性计算不符合论文（max+mean而非max+max）
3. KV cache压缩逻辑完全未实现

### ✅ 修复后的实现

1. **真正删除token**：使用索引选择，seq_len维度真正减小
2. **自动压缩KV cache**：在pruning_layer完成时自动压缩所有层
3. **符合论文**：按照PLLaVA实现和论文要求的max+max聚合

## 文件变更

### v2.0 新增文件
- `qwen_prunevid/stage1_utils.py` - Stage 1的clustering算法（DPC-KNN等）
- `qwen_prunevid/stage1_wrapper.py` - Stage 1 wrapper类
- `test_prunevid_stage1.py` - Stage 1参数扫描测试脚本

### v1.0 新增文件
- `qwen_prunevid/prunevid_cache.py` - PruneVidDynamicCache类（Stage 3核心实现）

### 修改文件
- `qwen_prunevid/model_wrapper.py` - 集成Stage 1，使用自定义cache
- `qwen_prunevid/qwen25_adapter.py` - 简化hook，集成自定义cache
- `qwen_prunevid/__init__.py` - 导出Stage 1相关类
- `test_prunevid_fix.py` - 添加超参数配置区，支持Stage 1测试

## 核心实现原理

### 1. PruneVidDynamicCache

继承自`DynamicCache`，override `update()`方法：

```python
def update(self, key_states, value_states, layer_idx, cache_kwargs):
    # 正常update
    keys, values = super().update(...)

    # 在pruning_layer时触发压缩
    if layer_idx == self.pruning_layer:
        # 构建kept_indices
        all_kept_indices = torch.cat([kept_visual_abs, text_indices])

        # 压缩所有已完成层的KV cache（真正删除token）
        for lid in range(layer_idx + 1):
            self.layers[lid].keys = old_keys[:, :, all_kept_indices, :].contiguous()
            self.layers[lid].values = old_values[:, :, all_kept_indices, :].contiguous()

    return keys, values
```

**关键点**：
- `tensor[:, :, indices, :]` 创建新的更小的tensor（真正减小seq_len）
- 压缩所有层（0到pruning_layer），后续层自动看到压缩后的cache

### 2. 简化的Hook逻辑

只需一个hook计算importance和indices：

```python
def compute_kept_indices_hook(module, input, output):
    # 1. 提取attention weights
    attention_weights = output[1]

    # 2. 计算importance（max+max，符合论文）
    text_to_visual = attention_weights[:, :, text_start:, visual_start:visual_end]
    importance = text_to_visual.max(dim=2)[0].max(dim=1)[0]

    # 3. Top-k选择
    _, topk_indices = torch.topk(importance, k=num_keep)

    # 4. 配置cache
    self.custom_cache.configure_pruning(
        pruning_layer=layer_idx,
        kept_visual_indices=topk_indices,
        visual_start=visual_start,
        visual_end=visual_end
    )
```

### 3. 集成到generate()

```python
# 创建自定义cache
past_key_values = self.adapter.create_custom_cache()

# Generate时传入
output = self.model.generate(
    **inputs,
    past_key_values=past_key_values,  # 自定义cache
    output_attentions=True,  # 必须！
    ...
)
```

## 使用方法

### 完整三阶段PruneVid（推荐）

```python
from qwen_prunevid import Qwen25VLPruneVid

# 创建model（启用Stage 1 + Stage 2）
model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    # Stage 1参数
    enable_stage1=True,
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
    # Stage 2参数
    enable_pruning=True,
    keep_ratio=0.4,
    pruning_layer=10,
    verbose=True
)

# 生成
result = model.generate(
    video_path="video.mp4",
    question="What is happening?",
    max_new_tokens=100
)

print(f"答案: {result['generated_text']}")
print(f"Stage 1: {result['tokens_before_stage1']} → {result['tokens_after_stage1']}")
print(f"Stage 2: {result['tokens_before']} → {result['tokens_after']}")
print(f"总压缩比: {result['total_compression_ratio']:.1%}")
```

### 仅Stage 2（v1.0方式）

```python
# 仅使用Stage 2 + Stage 3
model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    enable_stage1=False,  # 关闭Stage 1
    enable_pruning=True,
    keep_ratio=0.4,
    pruning_layer=10,
)

result = model.generate(
    video_path="video.mp4",
    question="What is happening?"
)
```

### 仅Stage 1

```python
# 仅使用Stage 1（测试vision encoder层面的压缩）
model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    enable_stage1=True,
    tau=0.8,
    cluster_ratio=0.5,
    temporal_segment_ratio=0.25,
    enable_pruning=False,  # 关闭Stage 2
)
```

### 验证效果

#### 基础测试

```bash
# 测试当前配置
python test_prunevid_fix.py video.mp4 "What is happening in the video?"
```

在`test_prunevid_fix.py`顶部修改`CURRENT_PRESET`切换配置：
- `'baseline'`: 无pruning
- `'stage2_only'`: 仅Stage 2
- `'stage1_only'`: 仅Stage 1
- `'default'`: Stage 1 + Stage 2（默认参数）
- `'conservative'`: 保守配置（优先精度）
- `'aggressive'`: 激进配置（最大压缩）
- `'custom'`: 自定义参数

#### Stage 1参数扫描

```bash
# 自动测试多种Stage 1参数组合
python test_prunevid_stage1.py video.mp4 "What is happening?"
```

会测试：
- tau: [0.7, 0.8, 0.9]
- cluster_ratio: [0.3, 0.5, 0.7]
- temporal_segment_ratio: [0.25, 0.5]

结果保存到`stage1_sweep_results.json`

### 预期效果

#### 完整三阶段（Stage 1 + Stage 2）

| 配置 | Stage 1压缩 | Stage 2压缩 | 总压缩 | 预期加速 | 精度影响 |
|------|------------|------------|--------|---------|---------|
| Conservative | ~20% | ~40% | ~52% | 1.5x | <2% |
| Default | ~50% | ~60% | ~80% | 2-3x | 2-3% |
| Aggressive | ~50% | ~70% | ~85% | 3-4x | 3-5% |

#### 仅Stage 2（v1.0）

| 指标 | 预期（keep_ratio=0.4） |
|------|----------------------|
| Token保留率 | ~40% |
| 压缩比 | ~60% |
| FLOPs减少 | 50-60% |
| TTFT加速 | 1.3-1.5x |
| 内存节省 | 40-50% |
| 准确率 | 与baseline相当 |

## 关键注意事项

### ⚠️ 必须启用output_attentions

```python
output = model.generate(
    ...,
    output_attentions=True,  # 必须！否则无法获取attention权重
)
```

没有attention权重时会fallback到基于norm的importance，效果可能不佳。

### ⚠️ Stage 1参数调优

**tau（静态/动态阈值）**：
- `tau=0.9`：严格，更少静态区域，保留更多动态信息
- `tau=0.8`：默认，平衡
- `tau=0.7`：宽松，更多静态区域，更高压缩

**cluster_ratio（空间聚类比例）**：
- `cluster_ratio=0.8`：保守，更少压缩
- `cluster_ratio=0.5`：默认，平衡
- `cluster_ratio=0.3`：激进，最大压缩

**temporal_segment_ratio（时序分段比例）**：
- `temporal_segment_ratio=0.5`：更少segments，每个更长
- `temporal_segment_ratio=0.25`：默认，平衡
- `temporal_segment_ratio=0.1`：更多segments，更精细的时序建模

**调优策略**：
1. 先运行`test_prunevid_stage1.py`进行参数扫描
2. 根据你的需求选择合适配置：
   - 优先精度：提高所有参数（0.9, 0.8, 0.5）
   - 平衡：默认参数（0.8, 0.5, 0.25）
   - 优先效率：降低参数（0.7, 0.3, 0.25）

### ⚠️ Stage 2参数调优

**keep_ratio（保留比例）**：
- `keep_ratio=0.6`：保守，更好的准确率
- `keep_ratio=0.4`：默认，平衡
- `keep_ratio=0.3`：激进，最大压缩

**pruning_layer（剪枝层）**：
- `pruning_layer=10`：默认（Qwen2.5-7B共32层）
- 太早（<8）：attention可能不稳定
- 太晚（>15）：节省的计算量减少

## 调试技巧

### 查看详细日志

```python
model = Qwen25VLPruneVid(..., verbose=True)
```

会打印：
- Visual token检测信息
- Attention shape
- Importance计算方式
- Pruning统计

### 检查cache shape

在hook中添加：

```python
print(f"Before pruning: {cache.key_cache[0].shape}")
# 应该在pruning后变小
```

### 对比baseline

```python
# Baseline
model_base = Qwen25VLPruneVid(..., enable_pruning=False)
result_base = model_base.generate(...)

# PruneVid
model_prune = Qwen25VLPruneVid(..., enable_pruning=True)
result_prune = model_prune.generate(...)

# 对比时间、内存、准确率
```

## 技术细节

### 为什么使用自定义Cache而不是Hook？

**Hook方案的问题**：
- 难以在hook中同时修改hidden_states、past_key_values、attention_mask等
- 需要协调多个hook，逻辑复杂
- 可能与transformers内部逻辑冲突

**自定义Cache的优势**：
- 在Cache层面统一处理，逻辑清晰
- 自动propagate到后续层
- 与transformers架构兼容性好

### 真正删除 vs Mask置零

```python
# ❌ 错误：Mask置零
mask = torch.zeros(seq_len)
mask[kept_indices] = 1.0
hidden_states = hidden_states * mask  # 序列长度仍是seq_len

# ✅ 正确：索引选择
hidden_states = hidden_states[:, kept_indices, :]  # 序列长度变为len(kept_indices)
```

Mask置零：
- 序列长度不变
- Attention仍计算所有位置（包括0）
- 无加速、无内存节省

索引选择：
- 序列长度真正减小
- Attention只计算保留的token
- 实际加速和内存节省

## 常见问题

### Q: 为什么我看不到加速？

检查：
1. `output_attentions=True` 是否启用
2. Verbose日志中pruning是否真的应用了
3. Cache shape是否真的变小了

### Q: 准确率下降怎么办？

调优：
1. 增大keep_ratio（如从0.4到0.5）
2. 调整pruning_layer（如从10到12）
3. 确认attention权重正确获取（不是fallback到norm）

### Q: 如何在EgoSchema上测试？

参考 `eval_qwen25_prunevid_egoschema.py`：

```python
model = Qwen25VLPruneVid(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    enable_pruning=True,
    keep_ratio=0.4,
    verbose=False  # 评估时关闭verbose
)

# 处理每个样本
prediction, text, stats = model.process_egoschema_sample(
    video_path, question, options
)
```

## 参考资料

- 论文：PruneVid: Visual Token Pruning for Efficient Video Large Language Models (ACL 2025)
- PLLaVA实现：`/mnt/ssd_ext/huggingface/prunevid/models/pllava/elastic_cache.py`
- Qwen2.5-VL文档：https://github.com/QwenLM/Qwen2.5-VL

## 贡献者

修复实现基于对论文和PLLaVA官方代码的深入研究。
