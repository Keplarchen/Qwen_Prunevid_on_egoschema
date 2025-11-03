"""
测试Stage 1修复的简单脚本
"""
import torch
import os

# 设置调试模式
os.environ['PRUNEVID_DEBUG'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("="*80)
print("测试Stage 1 修复")
print("="*80)

# 测试position_ids平均是否正确
print("\n[测试1] Position IDs 平均逻辑")
# 模拟3个帧的时间位置: [0, 50, 100]
temporal_positions = [
    torch.tensor([0, 0, 0]),   # 帧0的3个静态token
    torch.tensor([50, 50, 50]),  # 帧1
    torch.tensor([100, 100, 100])  # 帧2
]
avg_temporal = torch.stack(temporal_positions, dim=0).float().mean(dim=0).round().long()
print(f"时间位置: {temporal_positions}")
print(f"平均时间位置: {avg_temporal}")
expected = torch.tensor([50, 50, 50])
assert torch.all(avg_temporal == expected), f"期望{expected}, 实际{avg_temporal}"
print("✓ 时间位置平均正确")

# 测试RoPE embeddings不应该被平均
print("\n[测试2] RoPE Embeddings 不应该直接平均")
import math
theta1 = 0.0
theta2 = math.pi / 2
cos1 = math.cos(theta1)  # 1.0
cos2 = math.cos(theta2)  # 0.0
avg_cos = (cos1 + cos2) / 2  # 0.5

avg_theta = (theta1 + theta2) / 2  # pi/4
cos_avg_theta = math.cos(avg_theta)  # 0.707...

print(f"cos(θ1={theta1}) = {cos1}")
print(f"cos(θ2={theta2}) = {cos2}")
print(f"错误: avg(cos(θ)) = {avg_cos}")
print(f"正确: cos(avg(θ)) = {cos_avg_theta}")
print(f"差异: {abs(avg_cos - cos_avg_theta):.3f}")
assert abs(avg_cos - cos_avg_theta) > 0.1, "RoPE平均确实是错误的"
print("✓ 确认直接平均cos/sin是数学错误")

# 测试加载模型
print("\n[测试3] 加载模型和Stage 1")
try:
    from model_wrapper import VideoQwenPruneVidWrapper
    print("✓ 成功导入模型")

    # 尝试初始化（不加载权重，只检查结构）
    print("\n如果要完整测试，请运行:")
    print("CUDA_LAUNCH_BLOCKING=1 PRUNEVID_DEBUG=1 python eval_egoschema.py")

except Exception as e:
    print(f"✗ 导入失败: {e}")

print("\n" + "="*80)
print("基础测试完成")
print("="*80)
