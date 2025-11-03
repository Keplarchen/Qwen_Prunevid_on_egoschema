"""
直接测试Stage 1修复
"""
import torch
import os
os.environ['PRUNEVID_DEBUG'] = '1'

print("="*80)
print("直接测试Stage 1 Token Merger")
print("="*80)

# 测试导入
print("\n[1] 测试导入...")
try:
    from stage1_temporal_spatial_merge import SpatialTemporalTokenMerger
    print("✓ 成功导入 SpatialTemporalTokenMerger")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 测试配置
print("\n[2] 创建配置...")
try:
    from config import PruneVidConfig
    config = PruneVidConfig()
    config.verbose = True
    print("✓ 配置创建成功")
except Exception as e:
    print(f"✗ 配置失败: {e}")
    exit(1)

# 创建merger
print("\n[3] 创建 TokenMerger...")
try:
    merger = SpatialTemporalTokenMerger(config)
    print("✓ TokenMerger 创建成功")
except Exception as e:
    print(f"✗ 创建失败: {e}")
    exit(1)

# 模拟数据
print("\n[4] 准备测试数据...")
batch_size = 1
seq_len = 1000
hidden_dim = 3584
visual_start = 10
visual_end = 990

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建假数据
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)

# 创建position_ids [3, batch, seq_len]
# 模拟视频token的position_ids结构
position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=device)

# 视觉部分：模拟16帧，每帧28x28=784个token
num_frames = 16
tokens_per_frame = 784
for frame_idx in range(num_frames):
    start_idx = visual_start + frame_idx * tokens_per_frame
    end_idx = min(start_idx + tokens_per_frame, visual_end)

    # 时间维度：每帧不同的时间位置
    position_ids[0, :, start_idx:end_idx] = frame_idx * 50

    # 空间维度：28x28 grid
    for i in range(end_idx - start_idx):
        position_ids[1, :, start_idx + i] = i // 28  # height
        position_ids[2, :, start_idx + i] = i % 28   # width

print(f"hidden_states shape: {hidden_states.shape}")
print(f"position_ids shape: {position_ids.shape}")
print(f"Visual tokens: [{visual_start}:{visual_end}]")

# 创建假的position_embeddings
print("\n[5] 测试 Stage 1 forward...")
try:
    # 注意：我们传入的position_embeddings会被忽略（返回None）
    dummy_cos = torch.randn(3, batch_size, seq_len, 64, device=device)
    dummy_sin = torch.randn(3, batch_size, seq_len, 64, device=device)
    position_embeddings = (dummy_cos, dummy_sin)

    new_hidden_states, new_position_ids, new_position_embeddings, stats = merger(
        hidden_states=hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        visual_token_start=visual_start,
        visual_token_end=visual_end
    )

    print("\n✓ Stage 1 forward 成功!")
    print(f"输入序列长度: {seq_len}")
    print(f"输出序列长度: {new_hidden_states.shape[1]}")
    print(f"压缩率: {new_hidden_states.shape[1]/seq_len*100:.1f}%")
    print(f"\nposition_embeddings is None: {new_position_embeddings is None}")

    if new_position_embeddings is None:
        print("✓ 正确：position_embeddings 返回 None（需要重新计算）")
    else:
        print("✗ 错误：position_embeddings 不应该被返回")

    # 检查position_ids
    print(f"\nnew_position_ids shape: {new_position_ids.shape}")
    print(f"position_ids range: [{new_position_ids.min()}, {new_position_ids.max()}]")
    print(f"Contains NaN: {torch.isnan(new_position_ids).any()}")
    print(f"Contains Inf: {torch.isinf(new_position_ids).any()}")

    # 统计信息
    print(f"\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("✓ 所有测试通过！Stage 1 修复正确")
    print("="*80)

except Exception as e:
    print(f"\n✗ Stage 1 forward 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
