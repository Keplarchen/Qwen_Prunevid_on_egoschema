#!/usr/bin/env python3
"""
实时监控GPU使用情况
按 Ctrl+C 退出
"""
import subprocess
import time
import os
import sys

def clear_screen():
    """清空终端屏幕"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_gpu_info():
    """获取GPU使用信息"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return result.strip().split('\n')
    except Exception as e:
        return [f"错误: {e}"]

def get_gpu_processes():
    """获取GPU进程信息"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,used_memory,name',
             '--format=csv,noheader'],
            encoding='utf-8'
        )
        processes = result.strip()
        return processes if processes else None
    except:
        return None

def format_gpu_info(gpu_lines, show_processes=True):
    """格式化GPU信息"""
    print("=" * 100)
    print(f"{'GPU监控':^100}")
    print(f"{'实时更新中... (按 Ctrl+C 退出)':^100}")
    print("=" * 100)
    print()

    idle_gpus = []
    busy_gpus = []

    for line in gpu_lines:
        if '错误' in line:
            print(line)
            continue

        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue

        gpu_id = parts[0]
        gpu_name = parts[1]
        mem_used = float(parts[2]) if parts[2] not in ['N/A', ''] else 0.0
        mem_total = float(parts[3]) if parts[3] not in ['N/A', ''] else 0.0
        gpu_util = float(parts[4]) if parts[4] not in ['N/A', ''] else 0.0
        temp = parts[5] if parts[5] not in ['N/A', ''] else '0'

        # 功耗信息（如果有）
        power_draw = parts[6] if len(parts) > 6 and parts[6] not in ['N/A', ''] else '0'
        power_limit = parts[7] if len(parts) > 7 and parts[7] not in ['N/A', ''] else '0'

        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

        # 根据使用情况显示状态
        if gpu_util < 10 and mem_percent < 5:
            status = "🟢 空闲"
            idle_gpus.append(gpu_id)
        elif gpu_util < 30:
            status = "🟡 轻度使用"
            busy_gpus.append(gpu_id)
        elif gpu_util < 70:
            status = "🟠 中度使用"
            busy_gpus.append(gpu_id)
        else:
            status = "🔴 高负载"
            busy_gpus.append(gpu_id)

        print(f"{status} GPU {gpu_id}: {gpu_name}")
        print(f"    显存: {mem_used:>8.0f} MB / {mem_total:>8.0f} MB ({mem_percent:>5.1f}%)")
        print(f"    利用率: {gpu_util:>5.1f}%  |  温度: {temp}°C  |  功耗: {power_draw}W / {power_limit}W")
        print()

    # 显示空闲GPU列表
    print("-" * 100)
    if idle_gpus:
        print(f"✅ 空闲的GPU: {', '.join(idle_gpus)}")
    else:
        print(f"⚠️  没有空闲的GPU")

    if busy_gpus:
        print(f"⚙️  忙碌的GPU: {', '.join(busy_gpus)}")
    print("-" * 100)

    # 显示进程信息
    if show_processes:
        processes = get_gpu_processes()
        if processes:
            print("\n运行中的GPU进程:")
            print("-" * 100)
            print(processes)
            print("-" * 100)
        else:
            print("\n✓ 没有运行中的GPU进程")

    print()
    print("=" * 100)
    print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

def main():
    """主循环"""
    # 解析命令行参数
    update_interval = 2  # 默认每2秒更新一次
    show_once = False
    show_processes = True

    # 简单的参数解析
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--once':
                show_once = True
            elif arg == '--no-process':
                show_processes = False
            elif arg.startswith('--interval='):
                try:
                    update_interval = float(arg.split('=')[1])
                except:
                    print(f"警告: 无效的间隔参数 '{arg}'，使用默认值 2 秒")
            elif arg in ['-h', '--help']:
                print("GPU监控脚本")
                print("\n使用方法:")
                print("  python monitor_gpu.py                    # 默认每2秒刷新")
                print("  python monitor_gpu.py --interval=5       # 每5秒刷新")
                print("  python monitor_gpu.py --once             # 只显示一次")
                print("  python monitor_gpu.py --no-process       # 不显示进程信息")
                print("\n按 Ctrl+C 退出监控")
                sys.exit(0)

    try:
        if show_once:
            # 只显示一次，不清屏
            gpu_lines = get_gpu_info()
            print()
            format_gpu_info(gpu_lines, show_processes)
        else:
            # 持续监控
            while True:
                clear_screen()
                gpu_lines = get_gpu_info()
                format_gpu_info(gpu_lines, show_processes)
                time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止。")

if __name__ == "__main__":
    main()
