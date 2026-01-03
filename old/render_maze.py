#!/usr/bin/env python3
"""渲染迷宫布局的3D可视化工具。

读取 `generator.py` 生成的 JSON 文件，并使用matplotlib创建3D可视化图形。
用于快速预览生成的迷宫布局。
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# `generator.py` 输出的布局文件路径
JSON_PATH = Path("maze_layout.json")

# 各种轨道类型对应的颜色和符号
TRACK_STYLES = {
    "start": {"color": "green", "marker": "o", "size": 100, "label": "起点"},
    "end": {"color": "red", "marker": "s", "size": 100, "label": "终点"},
    "checkpoint": {"color": "yellow", "marker": "^", "size": 80, "label": "检查点"},
    "normal": {"color": "blue", "marker": ".", "size": 50, "label": "普通轨道"},
}


def get_track_type(name: str) -> str:
    """根据轨道名称返回轨道类型。
    
    Args:
        name: 轨道名称
        
    Returns:
        轨道类型字符串
    """
    lname = name.lower()
    if "start" in lname:
        return "start"
    if "end" in lname:
        return "end"
    if "checkpoint" in lname:
        return "checkpoint"
    return "normal"


def render_3d_maze() -> int:
    """读取 JSON 布局并创建3D可视化图形。
    
    Returns:
        0 表示成功，其他值表示错误
    """
    try:
        # 检查文件是否存在
        if not JSON_PATH.exists():
            print(f"错误：找不到文件 {JSON_PATH}")
            return 1
        
        # 读取 JSON 数据
        data = json.loads(JSON_PATH.read_text())
        
        if not data:
            print("没有轨道需要渲染。")
            return 0
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 按轨道类型分组数据
        track_groups = {}
        for item in data:
            track_type = get_track_type(item["name"])
            if track_type not in track_groups:
                track_groups[track_type] = []
            track_groups[track_type].append(item)
        
        # 绘制每种类型的轨道
        for track_type, tracks in track_groups.items():
            style = TRACK_STYLES[track_type]
            
            # 提取坐标
            xs = [track["position_maze"][0] for track in tracks]
            ys = [track["position_maze"][1] for track in tracks]
            zs = [track["position_maze"][2] for track in tracks]
            
            # 绘制散点图
            ax.scatter(xs, ys, zs, 
                      c=style["color"], 
                      marker=style["marker"], 
                      s=style["size"], 
                      label=style["label"],
                      alpha=0.8)
            
            # 为起点和终点添加文本标签
            if track_type in ["start", "end"]:
                for track in tracks:
                    x, y, z = track["position_maze"]
                    ax.text(x, y, z, f'{track_type.upper()}\n({x},{y},{z})', 
                           fontsize=8, ha='center')
        
        # 绘制轨道之间的连接线（简化版）
        if len(data) > 1:
            # 按照放置顺序连接轨道
            xs = [track["position_maze"][0] for track in data]
            ys = [track["position_maze"][1] for track in data]
            zs = [track["position_maze"][2] for track in data]
            
            ax.plot(xs, ys, zs, 'gray', alpha=0.5, linewidth=1, linestyle='--')
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_zlabel('Z 坐标')
        ax.set_title('迷宫3D布局可视化')
        
        # 添加图例
        ax.legend()
        
        # 设置坐标轴范围
        if data:
            all_x = [track["position_maze"][0] for track in data]
            all_y = [track["position_maze"][1] for track in data]
            all_z = [track["position_maze"][2] for track in data]
            
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            ax.set_zlim(min(all_z) - 1, max(all_z) + 1)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 显示统计信息
        print(f"\n=== 迷宫3D渲染统计 ===")
        print(f"总轨道数: {len(data)}")
        for track_type, tracks in track_groups.items():
            print(f"{TRACK_STYLES[track_type]['label']}: {len(tracks)} 个")
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        return 0
        
    except Exception as e:
        print(f"渲染过程中发生错误: {e}")
        return 1

def render_ascii_maze() -> int:
    """读取 JSON 布局并以 ASCII 形式绘制迷宫（2D俯视图）。
    
    Returns:
        0 表示成功，其他值表示错误
    """
    try:
        # 检查文件是否存在
        if not JSON_PATH.exists():
            print(f"错误：找不到文件 {JSON_PATH}")
            return 1
        
        # 读取 JSON 数据
        data = json.loads(JSON_PATH.read_text())
        
        if not data:
            print("没有轨道需要渲染。")
            return 0
        
        # 建立坐标到字符的映射（使用X-Y平面）
        pos_map = {}
        for item in data:
            x, y, z = item["position_maze"]
            track_type = get_track_type(item["name"])
            
            # 使用不同字符表示不同类型的轨道
            symbol_map = {"start": "S", "end": "E", "checkpoint": "C", "normal": "#"}
            symbol = symbol_map[track_type]
            
            pos_map[(x, y)] = f"{symbol}({z})"
        
        # 计算坐标范围
        xs = [p[0] for p in pos_map.keys()]
        ys = [p[1] for p in pos_map.keys()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        print("\n=== 迷宫ASCII渲染 (俯视图，括号内为Z坐标) ===")
        
        # 按从上到下、从左到右的顺序输出迷宫
        for y in range(max_y, min_y - 1, -1):
            row = []
            for x in range(min_x, max_x + 1):
                cell = pos_map.get((x, y), ".")
                row.append(f"{cell:>6}")  # 右对齐，宽度6
            print("".join(row))
        
        return 0
        
    except Exception as e:
        print(f"ASCII渲染过程中发生错误: {e}")
        return 1

def main() -> int:
    """主函数：提供3D和ASCII两种渲染选项。
    
    Returns:
        0 表示成功，其他值表示错误
    """
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # 检查是否在交互模式下运行
        if sys.stdin.isatty():
            print("=== 迷宫渲染工具 ===")
            print("1. 3D可视化 (推荐)")
            print("2. ASCII俯视图")
            print("3. 两种都显示")
            
            try:
                choice = input("请选择渲染方式 (1/2/3，默认为1): ").strip()
                if not choice:
                    choice = "1"
            except (KeyboardInterrupt, EOFError):
                print("\n使用默认3D渲染")
                choice = "1"
        else:
            # 非交互模式，默认使用3D渲染
            choice = "1"
            print("非交互模式，使用默认3D渲染")
    
    try:
        if choice == "1":
            return render_3d_maze()
        elif choice == "2":
            return render_ascii_maze()
        elif choice == "3":
            result1 = render_ascii_maze()
            result2 = render_3d_maze()
            return max(result1, result2)
        else:
            print("无效选择，使用默认3D渲染")
            return render_3d_maze()
            
    except KeyboardInterrupt:
        print("\n用户取消操作")
        return 0
    except Exception as e:
        print(f"程序执行错误: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
