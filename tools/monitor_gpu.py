#!/usr/bin/env python3
"""
GPU实时监控工具
功能：实时显示所有GPU的使用情况，包括显存、利用率、温度等信息
用法：python monitor_gpu.py
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
    """
    获取GPU使用信息

    返回:
        list: 每行包含一个GPU的信息字符串
    """
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ],
            encoding='utf-8'
        )
        return result.strip().split('\n')
    except FileNotFoundError:
        return ["错误: 未找到 nvidia-smi 命令，请确认已安装NVIDIA驱动"]
    except Exception as e:
        return [f"错误: {e}"]

def get_gpu_processes():
    """
    获取正在使用GPU的进程信息

    返回:
        dict: {gpu_id: [进程信息列表]}
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,used_memory', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        # 解析进程信息（这里简化处理）
        return {}
    except:
        return {}

def format_memory_bar(used, total, width=30):
    """
    创建显存使用进度条

    参数:
        used: 已使用显存(MB)
        total: 总显存(MB)
        width: 进度条宽度

    返回:
        str: 格式化的进度条字符串
    """
    if total == 0:
        return "[" + " " * width + "]"

    percent = used / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def get_status_emoji(mem_used):
    """
    根据显存使用量返回状态emoji

    参数:
        mem_used: 已使用显存(MB)

    返回:
        tuple: (emoji, 状态文字)
    """
    if mem_used < 1024:
        return "🟢", "空闲"
    elif mem_used < 10000:
        return "🟡", "轻度使用"
    else:
        return "🔴", "使用中"

def format_gpu_info(gpu_lines):
    """
    格式化并打印GPU信息

    参数:
        gpu_lines: GPU信息行列表
    """
    # 打印标题
    print("=" * 110)
    print(f"{'🖥️  GPU 实时监控':^115}")
    print(f"{'实时更新中... (按 Ctrl+C 退出)':^115}")
    print("=" * 110)
    print()

    # 检查是否有错误
    for line in gpu_lines:
        if '错误' in line:
            print(f"  ❌ {line}")
            return

    # 解析并显示每个GPU的信息
    for line in gpu_lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue

        # 解析数据
        gpu_id = parts[0]
        gpu_name = parts[1]
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        gpu_util = parts[4]
        temp = parts[5]

        # 功率信息（如果有）
        power_draw = parts[6] if len(parts) > 6 else "N/A"
        power_limit = parts[7] if len(parts) > 7 else "N/A"

        # 计算显存使用百分比
        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

        # 获取状态
        emoji, status = get_status_emoji(mem_used)

        # 打印GPU信息
        print(f"{emoji} GPU {gpu_id}: {gpu_name}")
        print(f"  └─ 状态: {status}")

        # 显存信息
        mem_bar = format_memory_bar(mem_used, mem_total)
        print(f"  └─ 显存: {mem_bar} {mem_used:>8.0f} MB / {mem_total:>8.0f} MB ({mem_percent:>5.1f}%)")

        # 利用率和温度
        util_bar = format_memory_bar(float(gpu_util), 100, 20)
        print(f"  └─ 利用率: {util_bar} {gpu_util:>3}%")
        print(f"  └─ 温度: {temp}°C", end="")

        # 功率信息
        if power_draw != "N/A" and power_limit != "N/A":
            try:
                power_percent = (float(power_draw) / float(power_limit)) * 100
                print(f"  |  功率: {power_draw} W / {power_limit} W ({power_percent:.1f}%)")
            except:
                print()
        else:
            print()

        print()

    # 打印底部信息
    print("=" * 110)
    print(f"  更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

def print_usage():
    """打印使用说明"""
    print("""
GPU监控工具使用说明
==================

用法:
    python monitor_gpu.py [选项]

选项:
    -h, --help      显示此帮助信息
    -i, --interval  设置更新间隔（秒），默认为2秒

示例:
    python monitor_gpu.py              # 使用默认设置
    python monitor_gpu.py -i 1         # 每秒更新一次
    python monitor_gpu.py --interval 5 # 每5秒更新一次

快捷键:
    Ctrl+C          退出监控

""")

def main():
    """主函数"""
    # 默认更新间隔（秒）
    update_interval = 2

    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            return
        elif sys.argv[1] in ['-i', '--interval']:
            if len(sys.argv) > 2:
                try:
                    update_interval = float(sys.argv[2])
                    if update_interval < 0.5:
                        print("⚠️  警告: 更新间隔太短，设置为0.5秒")
                        update_interval = 0.5
                except ValueError:
                    print("❌ 错误: 无效的时间间隔")
                    return
            else:
                print("❌ 错误: 请指定更新间隔")
                return

    # 主循环
    try:
        print(f"\n🚀 启动GPU监控... (更新间隔: {update_interval}秒)\n")
        time.sleep(1)

        while True:
            clear_screen()
            gpu_lines = get_gpu_info()
            format_gpu_info(gpu_lines)
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止。\n")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}\n")

if __name__ == "__main__":
    main()