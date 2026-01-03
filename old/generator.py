#!/usr/bin/env python3

import csv
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

CSV_PATH = Path("rail_config.csv")

@dataclass
class RailConfig:
    """轨道配置类，包含迷宫生成的各种参数"""
    min_difficulty: float = 0.0  # 降低最小难度以便于测试
    max_difficulty: float = 1.0  # 进一步降低最大难度以便于测试超过max_difficulty的情况
    build_range: Tuple[int, int, int] = (2, 2, 2)  # 建造范围 (x, y, z)
    safe_range: Tuple[int, int, int] = (2, 2, 2)      # 安全范围 (x, y, z)
    world: str = "default_world"                         # 轨道世界指定
    shape: str = "default_shape"                         # 轨道外形指定

class TrackManager:
    def __init__(self):
        self.tracks: List[Dict] = []

    def load_from_csv(self) -> bool:
        if not CSV_PATH.exists():
            print(f"错误: 找不到 {CSV_PATH}")
            self.tracks.clear()
            return False
        try:
            self.tracks.clear()
            with CSV_PATH.open("r", encoding="utf8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    track = self._parse_csv_row(row)
                    if track is not None:
                        self.tracks.append(track)
            print(f"从CSV加载了 {len(self.tracks)} 个轨道")
            return True
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            self.tracks.clear()
            return False

    def _parse_csv_row(self, row: Dict[str, str]) -> Dict:
        try:
            name = (row.get("Name", "") or "").strip()
            difficulty_str = (row.get("Difficulty", "0") or "0").strip()
            difficulty = float(difficulty_str) if difficulty_str else 0.0
            size = (
                int((row.get("SizeX", "1") or "1")),
                int((row.get("SizeY", "1") or "1")),
                int((row.get("SizeZ", "1") or "1")),
            )
            track_type = (row.get("Type", "normal") or "normal").strip().lower()
            
            # 为轨道生成唯一的index（使用名称的哈希值）
            index = hash(name) % (2**32)
            
            exits = []
            for i in range(1, 4):
                dir_val = (row.get(f"Exit{i}Dir", "") or "").strip()
                if not dir_val:
                    continue
                pos_str = (row.get(f"Exit{i}Pos", "0,0,0") or "0,0,0").strip()
                try:
                    pos = tuple(int(p.strip()) for p in pos_str.split(","))
                    if len(pos) != 3:
                        pos = (0, 0, 0)
                except Exception:
                    pos = (0, 0, 0)
                rot_str = (row.get(f"Exit{i}Rot", "[0]") or "[0]").strip()
                try:
                    rotations = json.loads(rot_str)
                    if isinstance(rotations, int):
                        rotations = [rotations]
                    if not isinstance(rotations, list):
                        rotations = [0]
                except Exception:
                    rotations = [0]
                direction_vector = self._direction_to_vector(dir_val)
                exits.append(
                    {
                        "name": name,
                        "direction": dir_val,
                        "direction_vector": direction_vector,
                        "pos": pos,
                        "rotations": rotations,
                    }
                )
            return {
                "name": name,
                "index": index,
                "difficulty": difficulty,
                "size": size,
                "exits": exits,
                "type": track_type,
                "entrances": [
                    {
                        "pos": (0, 0, 0),
                        "direction_vector": (1, 0, 0)  # 默认从X+方向进入
                    }
                ]
            }
        except Exception:
            return None

    def _direction_to_vector(self, direction: str) -> Tuple[int, int, int]:
        d = (direction or "").strip().upper()
        mapping = {
            "X+": (1, 0, 0),
            "X-": (-1, 0, 0),
            "Y+": (0, 1, 0),
            "Y-": (0, -1, 0),
            "Z+": (0, 0, 1),
            "Z-": (0, 0, -1),
        }
        return mapping.get(d, (1, 0, 0))

    def get_all_tracks(self) -> List[Dict]:
        return self.tracks

    def get_tracks_by_type(self, track_type: str) -> List[Dict]:
        return [t for t in self.tracks if t.get("type") == track_type]

class MazeGenerator:
    def __init__(self, config: RailConfig, track_manager: TrackManager):
        self.config = config
        self.track_manager = track_manager
        
        self.total_difficulty = 0.0
        self.placed_tracks: List[Dict] = []
        self.previous_track_difficulty = 0.0  # 记录上一个轨道的难度
        # 3D地图，使用字典存储，键为坐标元组，值为轨道状态信息
        self.maze_map: Dict[Tuple[int, int, int], Dict] = {}
        # 初始化地图状态
        for x in range(config.build_range[0]):
            for y in range(config.build_range[1]):
                for z in range(config.build_range[2]):
                    self.maze_map[(x, y, z)] = {
                        "stat": 0,  # 0=空闲, 1=被占用, 2=出入口
                        "track": None,  # 轨道信息（如果被占用）
                        "exit": None,  # 出入口信息（如果是出入口）
                    }
        
        # 下一个需要处理的轨道入口信息
        self.next_entries: Dict[str, Dict] = {
            "default": {
                "position": (0, 0, 0),  # 起始位置
                "direction": (1, 0, 0),  # 默认朝向X+方向
                "rotations": [0],       # 可用的旋转角度
            }
        }
        self.possible_rotations = [  # 可能的轨道旋转角度（绕Z轴）
            (1, 0, 0),   # 0度
            (0, 1, 0),   # 90度
            (-1, 0, 0),  # 180度
            (0, -1, 0),  # 270度
        ]

    def set_cell_occupied(self, pos: Tuple[int, int, int], track_info: Dict) -> None:
        """设置某个位置为被轨道占用"""
        if pos in self.maze_map:
            self.maze_map[pos]["stat"] = 1
            self.maze_map[pos]["track"] = track_info
            self.maze_map[pos]["exit"] = None

    def set_cell_exit(self, pos: Tuple[int, int, int], exit_info: Dict) -> None:
        """设置某个位置为出入口"""
        if pos in self.maze_map:
            self.maze_map[pos]["stat"] = 2
            self.maze_map[pos]["track"] = None
            self.maze_map[pos]["exit"] = exit_info

    def get_cell_status(self, pos: Tuple[int, int, int]) -> int:
        """获取某个位置的状态"""
        return self.maze_map.get(pos, {"stat": 0})["stat"]

    def is_position_valid(self, pos: Tuple[int, int, int]) -> bool:
        """检查某个位置是否有效（在建造范围内且未被占用）"""
        x, y, z = pos
        bx, by, bz = self.config.build_range

        # 检查是否越界
        if not (0 <= x < bx and 0 <= y < by and 0 <= z < bz):
            # print(f"  >> 验证失败: 位置 {pos} 超出建造范围 {self.config.build_range}")
            return False

        # 检查是否已被占用
        if self.get_cell_status(pos) != 0:
            # print(f"  >> 验证失败: 位置 {pos} 已被占用")
            return False

        # print(f"  >> 验证成功: 位置 {pos} 有效")
        return True

    def _log_track_placement(self, track_info: Dict):
        """通用日志记录函数，用于打印轨道放置的详细信息"""
        track = track_info["track"]
        position = track_info["position"]
        rotation = track_info["rotation"]
        track_type_name = track['type'].capitalize()

        print(f"\n=== 成功放置 {track_type_name} 轨道 ===")
        print(f"轨道名称: {track['name']}")
        print(f"放置位置: {position}")
        print(f"轨道大小: {track['size']}")
        print(f"旋转角度: {rotation}")
        print("出口信息:")
        if not track['exits']:
            print("  - 该轨道无出口")
        
        # 使用预先计算好的出口信息
        for idx, exit_calc in enumerate(track_info.get("exits_info", []), 1):
            exit_info = exit_calc["original"]
            exit_pos = exit_calc["calculated_pos"]
            rotated_dir = exit_calc["calculated_dir"]
            
            print(f"  出口 {idx}:")
            print(f"    - 相对位置: {exit_info['pos']}")
            print(f"    - 实际位置: {exit_pos}")
            print(f"    - 旋转后朝向: {rotated_dir}")

    def _place_track_on_map(self, track_to_place: Dict, position: Tuple[int, int, int], rotation: Tuple[int, int, int]) -> bool:
        """
        通用的轨道放置函数。负责验证、放置、记录和打印日志。
        """
        print(f"\n正在放置 {track_to_place['type']} 轨道: {track_to_place['name']}...")
        print(f"  >> 目标位置: {position}, 旋转: {rotation}")

        # 1. 验证轨道主体位置
        track_positions = self.calculate_track_positions(position, track_to_place["size"])
        print(f"  >> 计算占用位置: {track_positions}")
        for pos in track_positions:
            if not self.is_position_valid(pos):
                print(f"  >> 验证失败: 位置 {pos} 无效或已被占用。")
                return False

        # 2. 放置轨道
        track_info = {
            "track": track_to_place,
            "position": position,
            "rotation": rotation,
            "exits_info": []  # 存储计算好的出口信息，避免重复计算
        }
        for pos in track_positions:
            self.set_cell_occupied(pos, track_info)

        # 3. 设置出口
        for exit_info in track_to_place["exits"]:
            exit_pos = self.calculate_exit_position(
                position,
                exit_info["pos"],
                rotation
            )
            print(f"  >> 调试: 原始方向向量: {exit_info['direction_vector']}, 旋转: {rotation}")
            rotated_dir = self.rotate_direction(exit_info["direction_vector"], rotation)
            print(f"  >> 调试: 旋转后方向向量: {rotated_dir}")
            
            # 保存计算好的出口信息
            track_info["exits_info"].append({
                "original": exit_info,
                "calculated_pos": exit_pos,
                "calculated_dir": rotated_dir
            })
            
            self.set_cell_exit(exit_pos, {
                "position": exit_pos,
                "direction": rotated_dir,
                "rotations": exit_info["rotations"]
            })

        # 4. 更新状态和记录
        self.placed_tracks.append(track_info)
        # 修改难度计算公式：{当前轨道难度 * (上一节轨道难度*0.1 + 1)}
        base_difficulty = track_to_place['difficulty']
        difficulty_multiplier = (self.previous_track_difficulty * 0.1 + 1)
        calculated_difficulty = base_difficulty * difficulty_multiplier
        
        print(f"  >> 难度计算: {base_difficulty} * ({self.previous_track_difficulty} * 0.1 + 1) = {calculated_difficulty:.2f}")
        
        self.total_difficulty += calculated_difficulty
        self.previous_track_difficulty = base_difficulty

        # 5. 打印日志
        self._log_track_placement(track_info)

        return True

    def place_start_track(self) -> bool:
        """放置起始轨道"""
        start_tracks = self.track_manager.get_tracks_by_type("start")
        if not start_tracks:
            print("错误：没有找到起始轨道")
            return False

        start_track = random.choice(start_tracks)
        rail_rotation = random.choice(self.possible_rotations)

        x = random.randint(0, self.config.build_range[0] - start_track["size"][0])
        y = random.randint(0, self.config.build_range[1] - start_track["size"][1])
        z = random.randint(0, self.config.build_range[2] - start_track["size"][2])
        start_pos = (x, y, z)
        
        # 使用通用的放置函数来放置起始轨道
        return self._place_track_on_map(start_track, start_pos, rail_rotation)

    def calculate_track_positions(self, start_pos: Tuple[int, int, int], 
                                size: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """计算轨道占用的所有位置"""
        print(f"  >> 计算轨道大小: {size}")
        positions = []
        x, y, z = start_pos
        for dx in range(size[0]):
            for dy in range(size[1]):
                for dz in range(size[2]):
                    positions.append((x + dx, y + dy, z + dz))
        return positions

    def calculate_exit_position(self, track_pos: Tuple[int, int, int],
                              exit_offset: Tuple[int, int, int],
                              rotation: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """计算出口的实际位置（考虑旋转）"""
        # 对于0度旋转，直接加上偏移
        x, y, z = track_pos
        dx, dy, dz = exit_offset
        
        # 暂时简化：只处理0度旋转的情况
        return (x + dx, y + dy, z + dz)

    def rotate_direction(self, direction: Tuple[int, int, int],
                        rotation: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """旋转方向向量"""
        dx, dy, dz = direction
        rx, ry, _ = rotation
        if rx == 0 and ry == 0:  # 0度 - 无旋转
            return (dx, dy, dz)
        elif rx == 1 and ry == 0:  # 90度
            return (dy, -dx, dz)
        elif rx == 0 and ry == 1:  # 90度（另一种表示）
            return (dy, -dx, dz)
        elif rx == -1 and ry == 0:  # 180度
            return (-dx, -dy, dz)
        else:  # 270度或其他
            return (-dy, dx, dz)
    
    def calculate_maze_bounds(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """计算迷宫实际占用的最小立方体边界
        
        Returns:
            (min_pos, max_pos): 最小和最大坐标
        """
        if not self.placed_tracks:
            return (0, 0, 0), (0, 0, 0)
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for track_info in self.placed_tracks:
            position = track_info["position"]
            size = track_info["track"]["size"]
            
            # 计算轨道占用的所有位置
            for dx in range(size[0]):
                for dy in range(size[1]):
                    for dz in range(size[2]):
                        x, y, z = position[0] + dx, position[1] + dy, position[2] + dz
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        min_z = min(min_z, z)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        max_z = max(max_z, z)
        
        return (int(min_x), int(min_y), int(min_z)), (int(max_x), int(max_y), int(max_z))
    
    def center_maze(self) -> None:
        """将迷宫在建造空间内居中对齐"""
        if not self.placed_tracks:
            return
        
        min_pos, max_pos = self.calculate_maze_bounds()
        
        # 计算迷宫当前尺寸
        maze_size = (max_pos[0] - min_pos[0] + 1, 
                    max_pos[1] - min_pos[1] + 1, 
                    max_pos[2] - min_pos[2] + 1)
        
        # 计算居中偏移量
        build_range = self.config.build_range
        offset = ((build_range[0] - maze_size[0]) // 2 - min_pos[0],
                 (build_range[1] - maze_size[1]) // 2 - min_pos[1],
                 (build_range[2] - maze_size[2]) // 2 - min_pos[2])
        
        print(f"\n=== 迷宫居中对齐 ===")
        print(f"迷宫边界: {min_pos} 到 {max_pos}")
        print(f"迷宫尺寸: {maze_size}")
        print(f"建造范围: {build_range}")
        print(f"居中偏移: {offset}")
        
        # 如果不需要偏移，直接返回
        if offset == (0, 0, 0):
            print("迷宫已经居中，无需调整")
            return
        
        # 清空当前地图状态
        for pos in self.maze_map:
            self.maze_map[pos] = {
                "stat": 0,
                "track": None,
                "exit": None
            }
        
        # 更新所有轨道的位置
        for track_info in self.placed_tracks:
            old_position = track_info["position"]
            new_position = (old_position[0] + offset[0],
                          old_position[1] + offset[1],
                          old_position[2] + offset[2])
            track_info["position"] = new_position
            
            # 重新设置地图状态
            track_positions = self.calculate_track_positions(new_position, track_info["track"]["size"])
            for pos in track_positions:
                self.set_cell_occupied(pos, track_info)
            
            # 重新设置出口
            for exit_calc in track_info.get("exits_info", []):
                exit_info = exit_calc["original"]
                new_exit_pos = self.calculate_exit_position(
                    new_position,
                    exit_info["pos"],
                    track_info["rotation"]
                )
                exit_calc["calculated_pos"] = new_exit_pos
                self.set_cell_exit(new_exit_pos, {
                    "position": new_exit_pos,
                    "direction": exit_calc["calculated_dir"],
                    "rotations": exit_info["rotations"]
                })
        
        print(f"迷宫已成功居中对齐，所有轨道位置已更新")

def vector_to_direction(vector: Tuple[int, int, int]) -> str:
    """将方向向量转换为方向字符串"""
    direction_map = {
        (1, 0, 0): "X+",
        (-1, 0, 0): "X-",
        (0, 1, 0): "Y+",
        (0, -1, 0): "Y-",
        (0, 0, 1): "Z+",
        (0, 0, -1): "Z-"
    }
    return direction_map.get(vector, "UNKNOWN")

def export_maze_layout(generator: MazeGenerator) -> None:
    """将迷宫布局导出为JSON文件"""
    layout_data = []
    
    for i, track_info in enumerate(generator.placed_tracks):
        track = track_info["track"]
        position = track_info["position"]
        rotation = track_info["rotation"]
        
        # 计算轨道的实际入口方向
        # 对于起始轨道，入口方向就是其朝向的反方向
        # 对于其他轨道，需要根据连接逻辑确定
        entrance_direction_vector = (0, 0, 0)
        entrance_direction = "unknown"
        
        if i == 0:  # 起始轨道
            # 起始轨道朝向y-，所以入口方向也是y-（从y-方向进入）
            entrance_direction = "y-"
            entrance_direction_vector = (0, -1, 0)
        else:
            # 对于其他轨道，找到第一个入口的方向
            if track["entrances"]:
                first_entrance = track["entrances"][0]
                rotated_entrance_dir = generator.rotate_direction(
                    first_entrance["direction"], rotation
                )
                entrance_direction_vector = rotated_entrance_dir
                entrance_direction = vector_to_direction(rotated_entrance_dir)
        
        # 转换为UE坐标系
        # 每个迷宫单位 = 16cm，轨道原点位置需要考虑朝向
        # 轨道原点应该在轨道的中心位置
        base_x = position[0] * 16 + 8  # 轨道中心X坐标
        base_y = position[1] * 16 + 8  # 轨道中心Y坐标
        base_z = position[2] * 16      # Z坐标不变
        
        # 根据轨道朝向调整原点位置
        # 轨道原点应该在入口边缘的中心
        if entrance_direction == "Y-":
            # 从y-方向进入，原点在轨道的y+边缘中心
            position_cm = [base_x, base_y + 8, base_z]
        elif entrance_direction == "Y+":
            # 从y+方向进入，原点在轨道的y-边缘中心
            position_cm = [base_x, base_y - 8, base_z]
        elif entrance_direction == "X-":
            # 从x-方向进入，原点在轨道的x+边缘中心
            position_cm = [base_x + 8, base_y, base_z]
        elif entrance_direction == "X+":
            # 从x+方向进入，原点在轨道的x-边缘中心
            position_cm = [base_x - 8, base_y, base_z]
        else:
            # 默认使用中心位置
            position_cm = [base_x, base_y, base_z]
        
        # 将旋转角度转换为自旋值（0,1,2,3，每个对应90度）
        self_rotation = 0
        if rotation == (1, 0, 0):  # X+
            self_rotation = 0
        elif rotation == (0, 1, 0):  # Y+
            self_rotation = 1
        elif rotation == (-1, 0, 0):  # X-
            self_rotation = 2
        elif rotation == (0, -1, 0):  # Y-
            self_rotation = 3
        
        # 构建轨道条目
        track_entry = {
            "name": track["name"],
            "index": int(track.get("index", 0)),
            "position_maze": position,
            "position_cm": position_cm,
            "entrance_direction": entrance_direction,
            "self_rotation": self_rotation,
            "entrance_direction_vector": entrance_direction_vector,
            "difficulty": float(track["difficulty"])
        }
        
        layout_data.append(track_entry)
    
    # 保存到文件（不换行）
    with open("maze_layout.json", "w", encoding="utf8") as f:
        json.dump(layout_data, f, separators=(',', ':'))
    
    print(f"迷宫布局已导出到 maze_layout.json，共 {len(layout_data)} 个轨道")

def generate_maze(generator: MazeGenerator, max_tracks: int = 20, target_difficulty: float = None) -> bool:
    """生成完整的迷宫 - 使用连接式生成算法
    
    Args:
        generator: 迷宫生成器实例
        max_tracks: 最大轨道数量
        target_difficulty: 目标难度，如果为None则使用配置中的max_difficulty
    
    Returns:
        bool: 是否成功生成迷宫
    """
    # 如果未指定目标难度，则使用配置中的最大难度
    if target_difficulty is None:
        target_difficulty = generator.config.max_difficulty
    
    min_difficulty = generator.config.min_difficulty
    max_difficulty = generator.config.max_difficulty
    
    print(f"目标难度: {target_difficulty}")
    print(f"最小难度: {min_difficulty}")
    print(f"最大难度: {max_difficulty}")
    
    # 放置起始轨道在(0,0,0)，朝向y-
    start_tracks = generator.track_manager.get_tracks_by_type("start")
    if not start_tracks:
        print("错误：没有找到起始轨道")
        return False
    
    start_track = start_tracks[0]
    start_position = (0, 0, 0)
    start_rotation = (0, 0, 0)  # 朝向y-
    
    if not generator._place_track_on_map(start_track, start_position, start_rotation):
        print("起始轨道放置失败")
        return False
    
    print(f"起始轨道放置在 {start_position}，朝向 y-")
    
    # 获取所有普通轨道和终点轨道
    normal_tracks = generator.track_manager.get_tracks_by_type("normal")
    end_tracks = generator.track_manager.get_tracks_by_type("end")
    
    if not normal_tracks:
        print("错误：没有找到普通轨道")
        return False
    if not end_tracks:
        print("错误：没有找到终点轨道")
        return False
    
    # 按难度对轨道进行分组
    difficulty_groups = {}
    for track in normal_tracks:
        diff = float(track["difficulty"])
        difficulty_groups.setdefault(diff, []).append(track)
    
    difficulty_levels = sorted(difficulty_groups.keys())
    print(f"可用难度级别: {difficulty_levels}")
    
    # 连接式生成：从起始轨道的出口开始
    tracks_placed = 1
    current_exits = []  # 当前可用的出口列表
    
    # 计算起始轨道的出口
    for exit_info in start_track["exits"]:
         exit_pos = generator.calculate_exit_position(
             start_position, 
             exit_info["pos"], 
             start_rotation
         )
         exit_dir = generator.rotate_direction(
             exit_info["direction_vector"], 
             start_rotation
         )
         current_exits.append({
             "position": exit_pos,
             "direction": exit_dir,
             "from_track": 0  # 起始轨道索引
         })
    
    print(f"起始轨道出口: {current_exits}")
    
    # 主生成循环
    max_attempts = 100
    
    while (tracks_placed < max_tracks and 
           generator.total_difficulty < target_difficulty and 
           current_exits and 
           max_attempts > 0):
        
        # 选择一个出口进行连接
        exit_to_connect = random.choice(current_exits)
        current_exits.remove(exit_to_connect)
        
        # 计算下一个轨道的位置和入口方向
        next_position = exit_to_connect["position"]
        # 出口方向是轨道向外的方向，下一个轨道的入口应该是相反方向
        entrance_direction = exit_to_connect["direction"]  # 下一个轨道需要接收这个方向的连接
        
        print(f"\n尝试在位置 {next_position} 连接轨道，入口方向: {entrance_direction}")
        
        # 检查是否应该放置终点轨道
        should_place_end = False
        if generator.total_difficulty >= min_difficulty:
            if generator.total_difficulty >= max_difficulty:
                should_place_end = True
            else:
                probability = (generator.total_difficulty - min_difficulty) / (max_difficulty - min_difficulty)
                should_place_end = random.random() < probability * 0.3  # 降低概率
        
        placed = False
        
        # 选择要尝试的轨道类型
        if should_place_end:
            tracks_to_try = end_tracks
            print("尝试放置终点轨道")
        else:
            # 根据剩余难度选择合适的普通轨道
            remaining_difficulty = target_difficulty - generator.total_difficulty
            suitable_tracks = []
            
            for diff in difficulty_levels:
                if diff <= remaining_difficulty:
                    suitable_tracks.extend(difficulty_groups[diff])
            
            if not suitable_tracks:
                suitable_tracks = normal_tracks
            
            tracks_to_try = suitable_tracks
            print(f"尝试放置普通轨道，剩余难度: {remaining_difficulty:.2f}")
        
        # 尝试放置轨道
        for track in tracks_to_try:
             print(f"  >> 尝试轨道: {track['name']}")
             for rotation in generator.possible_rotations:
                 print(f"    >> 尝试旋转: {rotation}")
                 # 检查轨道是否有匹配的入口（使用出口信息反向匹配）
                 track_has_matching_entrance = False
                 
                 for exit_info in track["exits"]:
                     # 将出口方向反转作为入口方向
                     rotated_exit_dir = generator.rotate_direction(
                         exit_info["direction_vector"], rotation
                     )
                     reversed_entrance_dir = (-rotated_exit_dir[0], -rotated_exit_dir[1], -rotated_exit_dir[2])
                     print(f"      >> 出口方向: {exit_info['direction_vector']} -> 旋转后: {rotated_exit_dir} -> 入口方向: {reversed_entrance_dir}")
                     print(f"      >> 需要接收的方向: {entrance_direction}")
                     
                     # 检查这个轨道的入口是否能接收前一个轨道的出口
                     if reversed_entrance_dir == entrance_direction:
                         print(f"      >> 找到匹配的入口方向！")
                         track_has_matching_entrance = True
                         
                         # 计算轨道的实际放置位置（考虑入口偏移）
                         entrance_offset = generator.calculate_exit_position(
                             (0, 0, 0), 
                             exit_info["pos"], rotation
                         )
                         actual_position = (
                             next_position[0] - entrance_offset[0],
                             next_position[1] - entrance_offset[1],
                             next_position[2] - entrance_offset[2]
                         )
                         print(f"      >> 计算实际位置: 目标位置{next_position} - 入口偏移{entrance_offset} = {actual_position}")
                         
                         # 检查位置是否在建造范围内
                         if generator.is_position_valid(actual_position):
                             print(f"      >> 位置有效，尝试放置轨道...")
                             # 尝试放置轨道
                             if generator._place_track_on_map(track, actual_position, rotation):
                                 tracks_placed += 1
                                 placed = True
                                 print(f"成功放置轨道 {track['name']} 在 {actual_position}，旋转: {rotation}")
                                 
                                 # 添加新轨道的出口到可用出口列表
                                 for new_exit_info in track["exits"]:
                                     exit_pos = generator.calculate_exit_position(
                                         actual_position, 
                                         new_exit_info["pos"], 
                                         rotation
                                     )
                                     exit_dir = generator.rotate_direction(
                                         new_exit_info["direction_vector"], 
                                         rotation
                                     )
                                     current_exits.append({
                                         "position": exit_pos,
                                         "direction": exit_dir,
                                         "from_track": tracks_placed - 1
                                     })
                                 
                                 break
                 
                 if placed:
                     break
             
             if placed:
                 break
        
        if not placed:
            print(f"无法在位置 {next_position} 放置轨道")
            max_attempts -= 1
        
        # 如果放置了终点轨道，结束生成
        if placed and should_place_end:
            print("终点轨道放置成功，迷宫生成完成")
            break
    
    # 如果没有放置终点轨道，尝试在剩余出口放置
    if tracks_placed > 1 and current_exits:
        print("\n尝试在剩余出口放置终点轨道...")
        for exit_info in current_exits:
            next_position = exit_info["position"]
            entrance_direction = (-exit_info["direction"][0], 
                                -exit_info["direction"][1], 
                                -exit_info["direction"][2])
            
            for end_track in end_tracks:
                for rotation in generator.possible_rotations:
                    for entrance in end_track["entrances"]:
                        rotated_entrance_dir = generator.rotate_direction(
                            entrance["direction_vector"], rotation
                        )
                        
                        if rotated_entrance_dir == entrance_direction:
                            entrance_offset = generator.calculate_exit_position(
                                (0, 0, 0), entrance["pos"], rotation
                            )
                            actual_position = (
                                next_position[0] - entrance_offset[0],
                                next_position[1] - entrance_offset[1],
                                next_position[2] - entrance_offset[2]
                            )
                            
                            if (generator.is_position_valid(actual_position) and 
                                generator._place_track_on_map(end_track, actual_position, rotation)):
                                tracks_placed += 1
                                print(f"成功放置终点轨道在 {actual_position}")
                                break
                    if tracks_placed > len(generator.placed_tracks) - 1:
                        break
                if tracks_placed > len(generator.placed_tracks) - 1:
                    break
            if tracks_placed > len(generator.placed_tracks) - 1:
                break
    
    print(f"\n迷宫生成完成，共放置 {tracks_placed} 个轨道")
    print(f"最终难度: {generator.total_difficulty:.2f}")
    
    # 迷宫生成完成后进行居中对齐
    generator.center_maze()
    
    return True

def main():
    print("=== 迷宫生成器 - 轨道管理器测试 ===")
    manager = TrackManager()
    if not manager.load_from_csv():
        return

    tracks = manager.get_all_tracks()
    print("=== 已加载的轨道列表 ===")
    print("\n轨道统计:")
    by_type: Dict[str, List[Dict]] = {}
    for t in tracks:
        by_type.setdefault(t.get("type"), []).append(t)
    print(f"  总计: {len(tracks)} 个轨道")
    print("  按类型分布:")
    for k, v in by_type.items():
        if k in ["start", "end"]:
            print(f"    - {k}: {len(v)} 个 (特殊轨道)")
        else:
            print(f"    - {k}: {len(v)} 个")

    # 初始化迷宫生成器
    print("\n=== 迷宫生成器 - 初始化 ===")
    config = RailConfig()
    generator = MazeGenerator(config, manager)
    print(f"生成器初始化完成，建造范围: {config.build_range}")
    
    # 生成迷宫
    print("\n=== 开始生成迷宫 ===")
    if generate_maze(generator, max_tracks=30):  # 增加最大轨道数量，以便测试难度达到min_difficulty后的逻辑
        print("\n=== 迷宫生成完成 ===")
        print(f"放置轨道数量: {len(generator.placed_tracks)}")
        print(f"总难度: {generator.total_difficulty:.2f}")
        
        # 将迷宫布局导出为JSON
        export_maze_layout(generator)
    else:
        print("迷宫生成失败")

if __name__ == "__main__":
    main()
