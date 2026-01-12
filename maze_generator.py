import json
import random
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

# ==========================================
# 1. 基础配置与常量 (Basic Config & Constants)
# ==========================================

# 1.1 全局常量
# 定义比例关系: 1 逻辑格子 = 16 厘米
GRID_TO_WORLD_SCALE = 16.0 

# 迷宫最大尺寸 (逻辑单位 X, Y, Z)
# 扩大边界以避免早期死路，Z 轴通常为 1
MAZE_BOUNDS = None # 将在 Vector3 定义后初始化

# 目标生成参数
TARGET_DIFFICULTY = 15.0  # 目标总难度
TARGET_CHECKPOINTS = 0    # 目标检查点数量

# 1.2 边界模式枚举
# 模式0: 静态边界（默认）。迷宫必须在 (-MAZE_BOUNDS, +MAZE_BOUNDS) 的绝对坐标范围内生成。
# 模式1: 动态边界。迷宫的“尺寸”不能超过 (MAZE_BOUNDS * 2)，但绝对位置可以浮动。
BOUNDARY_MODE = 0

# ==========================================
# 2. 核心数据结构 (Core Data Structures)
# ==========================================

# 2.1 基础 3D 向量类
# 用于处理逻辑坐标和简单的向量运算。
@dataclass
class Vector3:
    x: int = 0
    y: int = 0
    z: int = 0

    # 2.1.1 转换为字典格式
    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    # 2.1.2 转换为世界坐标字典
    # 应用 GRID_TO_WORLD_SCALE
    def to_world_dict(self): 
        return {
            "x": float(round(self.x * GRID_TO_WORLD_SCALE, 8)), 
            "y": float(round(self.y * GRID_TO_WORLD_SCALE, 8)), 
            "z": float(round(self.z * GRID_TO_WORLD_SCALE, 8))
        }

    # 2.1.3 转换为元组格式
    def as_tuple(self):
        return (self.x, self.y, self.z)
    
    # 2.1.4 向量加法
    def add(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    # 2.1.5 Z 轴旋转逻辑
    # rot_index: 0=0度, 1=90度, 2=180度, 3=270度
    def rotate_z(self, rot_index):
        rx, ry = self.x, self.y
        # 顺时针旋转 -> 修正为逆时针/UE坐标系旋转 (0->X, 1->Y)
        for _ in range(rot_index % 4):
            # (x, y) -> (-y, x)
            rx, ry = -ry, rx
        return Vector3(rx, ry, self.z)

    # 2.1.6 X 轴旋转逻辑 (Roll)
    # rot_index: 0=0度, 1=90度, 2=180度, 3=270度
    def rotate_x(self, rot_index):
        ry, rz = self.y, self.z
        for _ in range(rot_index % 4):
            # (y, z) -> (-z, y)
            ry, rz = -rz, ry
        return Vector3(self.x, ry, rz)

# 初始化依赖 Vector3 的常量
MAZE_BOUNDS = Vector3(4, 4, 1)

# 2.2 轨道配置项
# 存储从 CSV 读取的静态配置信息。
@dataclass
class RailConfigItem:
    row_name: str
    diff_base: float
    size_rev: Vector3
    # 存储相对逻辑出口列表: 每个元素包含 Pos_Rev, Rot_Index_Offset, SpinDiff, LocalRot 等
    exits_logic: List[dict] 
    is_end: bool = False
    is_start: bool = False
    is_checkpoint: bool = False

# 2.3 开放连接点
# 表示迷宫生成过程中的待连接接口。
@dataclass
class OpenConnector:
    target_pos: Vector3     # 下一个轨道应该放置的逻辑坐标
    parent_id: int          # 父轨道的实例 Index
    parent_exit_idx: int    # 父轨道的出口索引
    accumulated_diff: float # 当前路径累积的难度
    
    # 旋转相关参数，用于动态计算目标旋转
    parent_rot_index: int
    parent_rot_abs: dict # 父级绝对旋转 (P, Y, R)
    spin_diffs: List[float]
    parent_exit_rot_offset: int = 0 # 父级出口的固有旋转偏移 (0-3)
    parent_exit_local_rot: dict = field(default_factory=lambda: {'p':0.0, 'y':0.0, 'r':0.0}) # 父级出口的局部旋转
    
    # 避免回退后重复尝试同一死路的候选黑名单
    forbidden_candidates: Set[str] = field(default_factory=set)

# 2.4 轨道实例
# 表示迷宫中已放置的一个具体轨道。
@dataclass
class RailInstance:
    rail_index: int
    rail_id: str
    pos_rev: Vector3    # 逻辑坐标
    rot_index: int      # 逻辑旋转 (0-3)
    rot_abs: dict       # 绝对旋转 (P, Y, R)
    size_rev: Vector3   # 逻辑尺寸
    diff_act: float     # 实际计算难度
    prev_index: int
    next_indices: List[int] = field(default_factory=list)
    # 存储出口连接状态 (用于 JSON 导出)
    exit_status: List[dict] = field(default_factory=list) 
    forbidden_siblings: Set[str] = field(default_factory=set) # 记录该节点生成时被禁用的兄弟节点 
    occupied_cells_rev: List[Vector3] = field(default_factory=list) # 存储实际占用的逻辑网格列表 

# ==========================================
# 3. 辅助计算逻辑 (Helper Logic)
# ==========================================

# 3.1 占用格子计算
# 根据轨道的 ID、位置、尺寸和旋转，计算其占用的所有逻辑格子。
def calculate_occupied_cells(rail_id: str, pos: Vector3, size: Vector3, rot_idx: int, roll_idx: int = 0) -> List[Tuple[int, int, int]]:
    # 3.1.1 确定局部遍历范围
    # 默认只占用 Forward (X) 方向，高度和宽度为 1 (即 0 偏移)
    # X 轴总是从 0 到 size.x - 1
    y_min, y_max = 0, 0
    z_min, z_max = 0, 0
    
    rid = rail_id.upper()
    
    # 3.1.2 解析 Y 轴范围
    if "_L90_" in rid or "_FL90_" in rid:
        # 向左延伸: [-(size.y-1), 0]
        y_min = -(size.y - 1)
        y_max = 0
    elif "_R90_" in rid or "_FR90_" in rid:
        # 向右延伸: [0, size.y-1]
        y_min = 0
        y_max = size.y - 1
    elif "_T_" in rid or "_CR_" in rid:
        # 对称延伸: [-(size.y-1), size.y-1]
        y_min = -(size.y - 1)
        y_max = size.y - 1
        
    # 3.1.3 解析 Z 轴范围
    if "_U90_" in rid or "_FU_" in rid:
        # 向上延伸: [0, size.z-1]
        z_min = 0
        z_max = size.z - 1
    elif "_D90_" in rid or "_FD_" in rid:
        # 向下延伸: [-(size.z-1), 0]
        z_min = -(size.z - 1)
        z_max = 0
        
    # 3.1.4 遍历并生成世界坐标
    cells = []
    
    for lx in range(size.x):
        for ly in range(y_min, y_max + 1):
            for lz in range(z_min, z_max + 1):
                # 构造局部坐标向量
                vec = Vector3(lx, ly, lz)
                
                # 应用 Roll (X轴旋转) - NEW
                rolled_vec = vec.rotate_x(roll_idx)
                
                # 应用 Yaw (Z轴旋转)
                rot_vec = rolled_vec.rotate_z(rot_idx)
                
                # 转换到世界/迷宫坐标
                gx = pos.x + rot_vec.x
                gy = pos.y + rot_vec.y
                gz = pos.z + rot_vec.z
                
                cells.append((gx, gy, gz))
                
    return cells

# ==========================================
# 4. 配置加载模块 (Config Loader)
# ==========================================

# 4.1 加载配置函数
# 从 CSV 文件加载轨道配置。
def load_config(csv_path):
    print(f"Loading Config from {csv_path}...")
    try:
        df_config = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        raise

    config_map = {}
    
    for _, row in df_config.iterrows():
        # 兼容旧版 Name 和新版 RowName
        name = row.get('RowName') if 'RowName' in row else row.get('Name')
        if pd.isna(name): continue
        
        # 兼容旧版 Difficulty 和新版 Diff_Base
        diff = 0.0
        if 'Diff_Base' in row and pd.notna(row['Diff_Base']):
            diff = float(row['Diff_Base'])
        elif 'Difficulty' in row and pd.notna(row['Difficulty']):
            diff = float(row['Difficulty'])
            
        # 解析尺寸 (Size)
        # 优先从列读取，否则从名称中解析 _X1_Y1_Z1
        sx, sy, sz = 1, 1, 1
        
        if 'SizeX' in row:
            sx = int(row['SizeX']) if pd.notna(row['SizeX']) else 1
            sy = int(row['SizeY']) if pd.notna(row['SizeY']) else 1
            sz = int(row['SizeZ']) if pd.notna(row['SizeZ']) else 1
        else:
            match_size = re.search(r"_X(\d+)_Y(\d+)_Z(\d+)", name)
            if match_size:
                sx, sy, sz = map(int, match_size.groups())

        size = Vector3(sx, sy, sz)
        
        # 解析出口 (Exits)
        exits = []
        
        # 新版: Exit_Array 字符串解析
        if 'Exit_Array' in row and pd.notna(row['Exit_Array']):
            exit_str = str(row['Exit_Array'])
            # 正则模式: Pos=(X=16,Y=0,Z=0),BaseRot=(P=0,Y=0,R=0)
            pat_pos = r"Pos=\(X=([\d.-]+),Y=([\d.-]+),Z=([\d.-]+)\)"
            pat_rot = r"BaseRot=\(P=([\d.-]+),Y=([\d.-]+),R=([\d.-]+)\)"
            
            pos_matches = list(re.finditer(pat_pos, exit_str))
            rot_matches = list(re.finditer(pat_rot, exit_str))
            
            for i, p_match in enumerate(pos_matches):
                px, py, pz = map(float, p_match.groups())
                
                # 将世界坐标转换为逻辑网格坐标
                gx = int(round(px / GRID_TO_WORLD_SCALE))
                gy = int(round(py / GRID_TO_WORLD_SCALE))
                gz = int(round(pz / GRID_TO_WORLD_SCALE))
                logic_pos = Vector3(gx, gy, gz)
                
                # 解析基础旋转 (BaseRot)
                rot_idx_offset = 0
                local_rot = {'p': 0.0, 'y': 0.0, 'r': 0.0}
                
                if i < len(rot_matches):
                    rp, ry, rr = map(float, rot_matches[i].groups())
                    local_rot = {'p': rp, 'y': ry, 'r': rr}
                    
                    rot_deg = int(ry)
                    # 归一化为 0-3 索引用于逻辑流
                    rot_idx_offset = int(rot_deg // 90) % 4
                
                exits.append({
                    "Pos": logic_pos,
                    "RotOffset": rot_idx_offset,
                    "LocalRot": local_rot,
                    "SpinDiff": [1.0, 1.0, 1.0, 1.0] # 默认 SpinDiff
                })

        # 旧版: Exit1Pos, Exit1Rot... (兼容逻辑)
        else:
            for i in range(1, 4):
                pos_col = f'Exit{i}Pos'
                rot_col = f'Exit{i}Rot'
                
                if pos_col in row and pd.notna(row[pos_col]):
                    pos_str = str(row[pos_col]).strip('"')
                    try:
                        px, py, pz = map(float, pos_str.split(','))
                        logic_pos = Vector3(int(px), int(py), int(pz))
                        
                        rot = int(row[rot_col]) if pd.notna(row[rot_col]) else 0
                        local_rot = {'p': 0.0, 'y': float(rot * 90), 'r': 0.0}
                        
                        exits.append({
                            "Pos": logic_pos,
                            "RotOffset": rot,
                            "LocalRot": local_rot,
                            "SpinDiff": [1.0, 1.0, 1.0, 1.0] 
                        })
                    except ValueError:
                        pass
        
        # 轨道类型判断
        rail_type = "normal"
        if 'Type' in row and pd.notna(row['Type']):
            rail_type = str(row['Type']).lower()
        else:
            if "start" in name.lower(): rail_type = "start"
            elif "end" in name.lower(): rail_type = "end"
            
        is_end = "end" in rail_type
        is_start = "start" in rail_type
        is_checkpoint = "checkpoint" in rail_type or "checkpoint" in name.lower()
        
        if is_start:
            print(f"Loaded Start Rail: {name}, Exits: {len(exits)}")
            if len(exits) == 0:
                 print(f"DEBUG: Exit_Array raw: {row.get('Exit_Array')}")

        config_map[name] = RailConfigItem(name, diff, size, exits, is_end, is_start, is_checkpoint)
        
    return config_map

# ==========================================
# 5. 迷宫生成器核心 (Maze Generator Core)
# ==========================================

class MazeGenerator:
    # 5.1 初始化 (Initialization)
    def __init__(self, config_map: Dict[str, RailConfigItem]):
        self.config_map = config_map
        self.placed_rails: List[RailInstance] = []
        self.occupied_cells: Dict[Tuple[int, int, int], int] = {}
        self.open_list: List[OpenConnector] = []
        self.global_index_counter = 0
        self.current_total_difficulty = 0.0
        
        # 检查点状态
        self.target_checkpoints = TARGET_CHECKPOINTS
        self.placed_checkpoints_count = 0
        self.segment_diff_acc = 0.0
        
        self.backtrack_count = 0

    # 5.2 状态检查与工具 (State & Utils)
    
    # 5.2.1 边界检查
    # 检查轨道是否在迷宫边界内。支持静态边界和动态边界模式。
    def is_in_bounds(self, pos: Vector3, occupied_cells: List[Tuple[int, int, int]]):
        # 计算当前组件的边界
        xs = [c[0] for c in occupied_cells]
        ys = [c[1] for c in occupied_cells]
        zs = [c[2] for c in occupied_cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        
        # print(f"DEBUG: Checking Bounds. Mode: {BOUNDARY_MODE}. Min/Max: X[{min_x},{max_x}], Y[{min_y},{max_y}], Z[{min_z},{max_z}]")
        
        if BOUNDARY_MODE == 0:
            # 模式0: 静态边界检查 [-Bound, Bound]
            if min_x < -MAZE_BOUNDS.x or max_x > MAZE_BOUNDS.x: 
                # print(f"DEBUG: OutOfBounds X: {min_x}, {max_x} vs {MAZE_BOUNDS.x}")
                return False
            if min_y < -MAZE_BOUNDS.y or max_y > MAZE_BOUNDS.y: 
                # print(f"DEBUG: OutOfBounds Y: {min_y}, {max_y} vs {MAZE_BOUNDS.y}")
                return False
            if min_z < -MAZE_BOUNDS.z or max_z > MAZE_BOUNDS.z: 
                # print(f"DEBUG: OutOfBounds Z: {min_z}, {max_z} vs {MAZE_BOUNDS.z}")
                return False 
            return True
            
        elif BOUNDARY_MODE == 1:
            # 模式1: 动态边界检查 (检查包围盒跨度)
            
            # 1. 获取当前全局边界
            if not hasattr(self, 'global_bounds'):
                curr_min_x, curr_max_x = float('inf'), float('-inf')
                curr_min_y, curr_max_y = float('inf'), float('-inf')
                curr_min_z, curr_max_z = float('inf'), float('-inf')
            else:
                curr_min_x, curr_max_x = self.global_bounds[0], self.global_bounds[1]
                curr_min_y, curr_max_y = self.global_bounds[2], self.global_bounds[3]
                curr_min_z, curr_max_z = self.global_bounds[4], self.global_bounds[5]
            
            # 2. 更新边界
            if curr_min_x == float('inf'):
                new_min_x, new_max_x = min_x, max_x
                new_min_y, new_max_y = min_y, max_y
                new_min_z, new_max_z = min_z, max_z
            else:
                new_min_x = min(curr_min_x, min_x)
                new_max_x = max(curr_max_x, max_x)
                new_min_y = min(curr_min_y, min_y)
                new_max_y = max(curr_max_y, max_y)
                new_min_z = min(curr_min_z, min_z)
                new_max_z = max(curr_max_z, max_z)
            
            # 3. 检查跨度 (Max - Min + 1)
            allowed_span_x = MAZE_BOUNDS.x * 2 + 1
            allowed_span_y = MAZE_BOUNDS.y * 2 + 1
            allowed_span_z = MAZE_BOUNDS.z * 2 + 1
            
            if (new_max_x - new_min_x + 1) > allowed_span_x: return False
            if (new_max_y - new_min_y + 1) > allowed_span_y: return False
            if (new_max_z - new_min_z + 1) > allowed_span_z: 
                # print(f"DEBUG: OutOfBounds Span Z: {new_max_z - new_min_z + 1} > {allowed_span_z}")
                return False
            
            return True
            
        return False

    # 5.2.2 碰撞检查
    # 检查给定的占用格子列表是否与已占用的格子冲突。
    def is_colliding(self, occupied_cells: List[Tuple[int, int, int]]):
        for cell in occupied_cells:
            if cell in self.occupied_cells:
                return True, self.occupied_cells[cell]
        return False, None

    # 5.2.3 占用标记
    # 将一组格子标记为被特定轨道占用，并更新全局边界(如果需要)。
    def mark_occupied(self, cells: List[Tuple[int, int, int]], rail_index: int):
        # 模式1下更新全局边界
        if BOUNDARY_MODE == 1:
            if not hasattr(self, 'global_bounds'):
                self.global_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
            
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            zs = [c[2] for c in cells]
            
            self.global_bounds[0] = min(self.global_bounds[0], min(xs))
            self.global_bounds[1] = max(self.global_bounds[1], max(xs))
            self.global_bounds[2] = min(self.global_bounds[2], min(ys))
            self.global_bounds[3] = max(self.global_bounds[3], max(ys))
            self.global_bounds[4] = min(self.global_bounds[4], min(zs))
            self.global_bounds[5] = max(self.global_bounds[5], max(zs))

        for cell in cells:
            self.occupied_cells[cell] = rail_index

    # 5.2.4 方向转换
    # 将旋转索引转换为 UE 方向字符串。
    # 0=+X, 1=+Y, 2=-X, 3=-Y
    def get_ue_dir_str(self, rot_idx):
        dirs = ["+X", "+Y", "-X", "-Y"]
        return dirs[rot_idx % 4]

    # 5.3 核心放置逻辑 (Core Placement Logic)
    
    # 5.3.1 计算旋转与变换 (Rotation & Transform Calculation)
    # 根据父级出口和候选轨道类型，计算新轨道的逻辑旋转索引和绝对旋转(P,Y,R)。
    # 使用简单的欧拉角分量加法：ParentRot + ExitLocalRot + Spin。
    # 修正：Spin 对应 Roll (Local X)，RotIdx (Yaw) 仅由 Parent+Exit 决定。
    def _calculate_rail_transform(self, rail_id: str, connector: OpenConnector, spin_rot: int) -> Tuple[int, dict, int]:
        # 1. 计算基础逻辑旋转 (Yaw Index 0-3)
        # 绝对旋转 = (ParentRot + ExitRotOffset) % 4 (NO SPIN in YAW)
        rot_idx = (connector.parent_rot_index + connector.parent_exit_rot_offset) % 4
        
        # 2. 计算 Roll 索引
        roll_idx = spin_rot
        
        # 3. 计算绝对旋转 (Rot_Abs: P, Y, R)
        # 简单的分量加法: Child = Parent + Exit + Spin
        
        p_parent = connector.parent_rot_abs.get('p', 0.0)
        y_parent = connector.parent_rot_abs.get('y', 0.0)
        r_parent = connector.parent_rot_abs.get('r', 0.0)
        
        p_exit = connector.parent_exit_local_rot.get('p', 0.0)
        y_exit = connector.parent_exit_local_rot.get('y', 0.0)
        r_exit = connector.parent_exit_local_rot.get('r', 0.0)
        
        # Spin 影响 Roll
        r_spin = float(spin_rot * 90.0)
        
        final_p = (p_parent + p_exit) % 360.0
        final_y = (y_parent + y_exit) % 360.0 # Yaw 不受 Spin 影响
        final_r = (r_parent + r_exit + r_spin) % 360.0
        
        rot_abs = {
            "p": final_p,
            "y": final_y,
            "r": final_r
        }
        
        return rot_idx, rot_abs, roll_idx

    # 5.3.2 放置函数占位
    def place_rail(self, rail_id: str, connector: OpenConnector = None, is_start=False):
        # 占位符，实际使用 place_rail_v2
        pass

    # 5.3.3 核心放置实现
    # 尝试在指定位置放置轨道。
    # 执行占用检查、碰撞检查、边界检查，若成功则实例化并更新状态。
    def place_rail_v2(self, rail_id: str, pos: Vector3, rot: int, rot_abs: dict, diff_base_acc: float, ratio: float, prev_idx: int, roll: int = 0):
        cfg = self.config_map[rail_id]
        
        # 1. 计算预期占用空间 (调用辅助计算逻辑)
        # Pass roll (Spin)
        expected_cells_tuples = calculate_occupied_cells(rail_id, pos, cfg.size_rev, rot, roll)
        
        # 2. 检查碰撞
        colliding, conflict_id = self.is_colliding(expected_cells_tuples)
        if colliding: 
            return f"Collision with Rail {conflict_id}"
            
        # 3. 检查边界
        if not self.is_in_bounds(pos, expected_cells_tuples): 
            return "OutOfBounds"
        
        # 4. 成功放置 -> 实例化
        
        # 计算实际难度
        current_diff = (1.0 + diff_base_acc * 0.1) * cfg.diff_base * ratio
        
        # 分配索引
        idx = self.global_index_counter
        self.global_index_counter += 1
        
        # 转换占用格子为 Vector3 列表存储
        occupied_cells_vecs = [Vector3(c[0], c[1], c[2]) for c in expected_cells_tuples]
        
        instance = RailInstance(
            rail_index=idx,
            rail_id=rail_id,
            pos_rev=pos,
            rot_index=rot,
            rot_abs=rot_abs, # 存储绝对旋转
            size_rev=cfg.size_rev, 
            diff_act=current_diff,
            prev_index=prev_idx,
            occupied_cells_rev=occupied_cells_vecs
        )
        
        # 初始化出口状态
        instance.exit_status = [{"Index": i, "IsConnected": False, "TargetID": -1, "WorldPos": None} for i in range(len(cfg.exits_logic))]

        # 更新全局状态
        self.mark_occupied(expected_cells_tuples, idx)
        self.placed_rails.append(instance)
        self.current_total_difficulty += current_diff

        # 将新出口加入 OpenList
        # print(f"DEBUG: Processing Exits for {rail_id}. Count: {len(cfg.exits_logic)}")
        for i, exit_data in enumerate(cfg.exits_logic):
            local_pos = exit_data['Pos']
            
            # 1. Apply Roll first
            rolled_offset = local_pos.rotate_x(roll)
            
            # 2. Apply Yaw
            rotated_offset = rolled_offset.rotate_z(rot) # 逻辑旋转
            
            world_exit_pos = pos.add(rotated_offset) # 世界逻辑坐标
            
            spin_diffs = exit_data['SpinDiff']
            exit_rot_offset = exit_data.get('RotOffset', 0)
            local_rot = exit_data.get('LocalRot', {'p':0.0, 'y':0.0, 'r':0.0}) # 获取局部旋转
            
            self.open_list.append(OpenConnector(
                target_pos=world_exit_pos,
                parent_id=idx,
                parent_exit_idx=i,
                accumulated_diff=current_diff,
                parent_rot_index=rot,
                parent_rot_abs=rot_abs, # 传递当前轨道的绝对旋转
                spin_diffs=spin_diffs,
                parent_exit_rot_offset=exit_rot_offset,
                parent_exit_local_rot=local_rot # 传递局部旋转
            ))
                    
        return instance

    # 5.4 生成流程 (Generation Flow)
    
    # 5.4.1 生成主循环
    # 迷宫生成主循环。
    def generate(self):
        print(f"Start Generating... Target Diff: {TARGET_DIFFICULTY}")
        
        # 1. 放置起点 (Start Rail)
        start_candidates = [k for k, v in self.config_map.items() if v.is_start]
        if not start_candidates: raise Exception("No Start Rail defined!")
        
        start_id = random.choice(start_candidates)
        print(f"Placing Start: {start_id}")
        
        # 在允许范围内随机放置起点
        start_cfg = self.config_map[start_id]
        start_sz = start_cfg.size_rev
        
        # 计算起点的合法边界
        min_x, max_x = -MAZE_BOUNDS.x, MAZE_BOUNDS.x - start_sz.x + 1
        min_y, max_y = -MAZE_BOUNDS.y, MAZE_BOUNDS.y - start_sz.y + 1
        
        if min_x > max_x: min_x = max_x = -MAZE_BOUNDS.x 
        if min_y > max_y: min_y = max_y = -MAZE_BOUNDS.y

        start_x = random.randint(min_x, max_x)
        start_y = random.randint(min_y, max_y)
        
        start_pos = Vector3(start_x, start_y, 0)
        start_rot = 0
        print(f"-> 尝试放置 Start [{start_id}] at Pos={start_pos.as_tuple()}, Dir_Abs={self.get_ue_dir_str(start_rot)}")
        
        # 起点默认旋转
        start_rot_abs = {"p": 0.0, "y": 0.0, "r": 0.0}
        start_res = self.place_rail_v2(start_id, start_pos, start_rot, start_rot_abs, 0, 1.0, -1, roll=0)
        if isinstance(start_res, str):
            raise Exception(f"Start Rail Placement Failed: {start_res}")
        # print(f"Start Rail Placed. OpenList Size: {len(self.open_list)}")
        
        # 计算分段难度目标
        segment_target_diff = TARGET_DIFFICULTY / (self.target_checkpoints + 1)
        print(f"Segment Target Diff: {segment_target_diff}")

        # 2. 生成主循环
        while True:
            # 检查是否达到目标难度
            must_end = self.current_total_difficulty >= TARGET_DIFFICULTY
            
            # 检查是否需要触发 Checkpoint
            trigger_checkpoint = (self.placed_checkpoints_count < self.target_checkpoints) and \
                                 (self.segment_diff_acc >= segment_target_diff)
            
            # 3. 处理死路与回退 (Backtracking)
            if not self.open_list:
                if self.placed_rails:
                    self.backtrack_count += 1
                    
                    # 3.1 移除最后一个轨道
                    last_rail = self.placed_rails.pop()
                    self.global_index_counter -= 1
                    self.current_total_difficulty -= last_rail.diff_act
                    
                    # 3.2 释放占用格子
                    cells_to_remove = calculate_occupied_cells(last_rail.rail_id, last_rail.pos_rev, last_rail.size_rev, last_rail.rot_index)
                    for cell in cells_to_remove:
                        if cell in self.occupied_cells and self.occupied_cells[cell] == last_rail.rail_index:
                            del self.occupied_cells[cell]
                    
                    # 3.3 恢复父级连接点
                    if last_rail.prev_index != -1:
                        parent = next((r for r in self.placed_rails if r.rail_index == last_rail.prev_index), None)
                        if parent:
                            for exit_idx, status in enumerate(parent.exit_status):
                                if status['TargetID'] == last_rail.rail_index:
                                    status['IsConnected'] = False
                                    status['TargetID'] = -1
                                    
                                    # 重新构建 OpenConnector
                                    parent_cfg = self.config_map[parent.rail_id]
                                    exit_data = parent_cfg.exits_logic[exit_idx]
                                    
                                    local_pos = exit_data['Pos']
                                    rotated_offset = local_pos.rotate_z(parent.rot_index)
                                    world_exit_pos = parent.pos_rev.add(rotated_offset)
                                    
                                    spin_diffs = exit_data['SpinDiff']
                                    exit_rot_offset = exit_data.get('RotOffset', 0)
                                    
                                    # 继承并添加禁忌表
                                    new_forbidden = last_rail.forbidden_siblings.copy()
                                    new_forbidden.add(last_rail.rail_id)

                                    local_rot = exit_data.get('LocalRot', {'p':0.0, 'y':0.0, 'r':0.0})

                                    self.open_list.append(OpenConnector(
                                        target_pos=world_exit_pos,
                                        parent_id=parent.rail_index,
                                        parent_exit_idx=exit_idx,
                                        accumulated_diff=self.current_total_difficulty,
                                        parent_rot_index=parent.rot_index,
                                        parent_rot_abs=parent.rot_abs,
                                        spin_diffs=spin_diffs,
                                        parent_exit_rot_offset=exit_rot_offset,
                                        parent_exit_local_rot=local_rot,
                                        forbidden_candidates=new_forbidden
                                    ))
                                    break
                    continue # 回退后立即重试
                
                break # 彻底无解

            # 4. 选择连接点
            connector_idx = random.randint(0, len(self.open_list) - 1)
            connector = self.open_list.pop(connector_idx)
            
            # 5. 筛选候选轨道
            if must_end:
                candidates = [k for k, v in self.config_map.items() if v.is_end]
            elif trigger_checkpoint:
                # Checkpoint 阶段 1: 必须放置分叉路口 (Exits >= 2)
                candidates = [k for k, v in self.config_map.items() 
                              if not v.is_end and not v.is_start and not v.is_checkpoint and len(v.exits_logic) >= 2]
                if not candidates:
                    print("Warning: No Fork Rails available for Checkpoint placement! Skipping Checkpoint logic this step.")
                    candidates = [k for k, v in self.config_map.items() if not v.is_end and not v.is_start and not v.is_checkpoint]
            else:
                candidates = [k for k, v in self.config_map.items() if not v.is_end and not v.is_start and not v.is_checkpoint]
            
            # 过滤禁忌候选
            original_candidate_count = len(candidates)
            candidates = [c for c in candidates if c not in connector.forbidden_candidates]
            if original_candidate_count > 0 and not candidates:
                print(f"All candidates forbidden for this connector (Tried: {connector.forbidden_candidates}). Skipping.")
                pass

            success = False
            attempts = 0
            placed_id = None
            placed_instance = None
            final_rot = 0
            fail_reasons = {}
            
            # 6. 计算所有可能的旋转 (TargetRot, Ratio)
            # 修改: 直接调用 _calculate_rail_transform 获取 rot_idx 和 rot_abs
            # 但这里我们是先遍历 spin_rot，然后在循环中对每个 candidate 计算 transform
            # 为了保持逻辑，我们先收集 (spin_rot, ratio)，然后在尝试 candidate 时计算具体 transform
            spin_options = []
            for spin_rot, ratio in enumerate(connector.spin_diffs):
                if ratio > 0:
                    spin_options.append((spin_rot, ratio))

            # 7. 尝试放置
            while candidates:
                cand_id = random.choice(candidates)
                candidates.remove(cand_id)
                
                rail_success = False
                
                for spin_rot, ratio in spin_options:
                    attempts += 1
                    
                    # 6.1 计算变换
                    target_rot, target_rot_abs, target_roll = self._calculate_rail_transform(cand_id, connector, spin_rot)
                    
                    # 尝试放置 API 调用
                    result = self.place_rail_v2(cand_id, connector.target_pos, target_rot, target_rot_abs, connector.accumulated_diff, ratio, connector.parent_id, roll=target_roll)
                    
                    if isinstance(result, RailInstance):
                        # 放置成功，更新父级
                        parent = next(r for r in self.placed_rails if r.rail_index == connector.parent_id)
                        parent.next_indices.append(result.rail_index)
                        parent.exit_status[connector.parent_exit_idx]['IsConnected'] = True
                        parent.exit_status[connector.parent_exit_idx]['TargetID'] = result.rail_index
                        
                        placed_id = cand_id
                        placed_instance = result
                        placed_instance.forbidden_siblings = connector.forbidden_candidates.copy()
                        final_rot = target_rot
                        
                        # Checkpoint 阶段 2: 尝试在分叉后立即放置 Checkpoint
                        if trigger_checkpoint:
                            checkpoint_placed_success = False
                            checkpoint_candidates = [k for k, v in self.config_map.items() if v.is_checkpoint]
                            
                            if not checkpoint_candidates:
                                success = True
                                rail_success = True
                                break

                            # 遍历新放置的分叉的所有出口
                            for exit_idx, status in enumerate(placed_instance.exit_status):
                                if status['IsConnected']: continue
                                
                                parent_cfg = self.config_map[placed_instance.rail_id]
                                exit_data = parent_cfg.exits_logic[exit_idx]
                                
                                local_pos = exit_data['Pos']
                                rotated_offset = local_pos.rotate_z(final_rot)
                                world_exit_pos = placed_instance.pos_rev.add(rotated_offset)
                                
                                spin_diffs = exit_data['SpinDiff']
                                exit_rot_offset = exit_data.get('RotOffset', 0)
                                parent_rot_idx = final_rot
                                local_rot = exit_data.get('LocalRot', {'p':0.0, 'y':0.0, 'r':0.0})
                                
                                # 构造临时 Connector 用于计算 Checkpoint 变换
                                temp_connector = OpenConnector(
                                    target_pos=world_exit_pos,
                                    parent_id=placed_instance.rail_index,
                                    parent_exit_idx=exit_idx,
                                    accumulated_diff=placed_instance.diff_act + connector.accumulated_diff,
                                    parent_rot_index=parent_rot_idx,
                                    parent_rot_abs=placed_instance.rot_abs, # 传递父级绝对旋转
                                    spin_diffs=spin_diffs,
                                    parent_exit_rot_offset=exit_rot_offset,
                                    parent_exit_local_rot=local_rot
                                )

                                # 尝试所有 Checkpoint 候选
                                for cp_id in checkpoint_candidates:
                                    for cp_spin_rot, spin_ratio in enumerate(spin_diffs):
                                        if spin_ratio <= 0: continue
                                        
                                        cp_target_rot, cp_target_rot_abs, cp_roll = self._calculate_rail_transform(cp_id, temp_connector, cp_spin_rot)
                                        
                                        cp_result = self.place_rail_v2(cp_id, world_exit_pos, cp_target_rot, cp_target_rot_abs,
                                                                     placed_instance.diff_act + connector.accumulated_diff,
                                                                     spin_ratio, placed_instance.rail_index, roll=cp_roll)
                                        
                                        if isinstance(cp_result, RailInstance):
                                            print(f"  -> Checkpoint Placed: {cp_id} at Exit {exit_idx}")
                                            
                                            placed_instance.next_indices.append(cp_result.rail_index)
                                            placed_instance.exit_status[exit_idx]['IsConnected'] = True
                                            placed_instance.exit_status[exit_idx]['TargetID'] = cp_result.rail_index
                                            
                                            self.placed_checkpoints_count += 1
                                            self.segment_diff_acc = 0.0
                                            checkpoint_placed_success = True
                                            break 
                                    if checkpoint_placed_success: break
                                if checkpoint_placed_success: break
                            
                            if checkpoint_placed_success:
                                success = True
                                rail_success = True
                                break
                            else:
                                # Checkpoint 放置失败，回滚分叉
                                print(f"  -> Failed to place Checkpoint on Fork {cand_id}, Rolling back Fork.")
                                self.placed_rails.pop()
                                self.global_index_counter -= 1
                                self.current_total_difficulty -= placed_instance.diff_act
                                
                                # 释放占用
                                cells_to_remove = calculate_occupied_cells(placed_instance.rail_id, placed_instance.pos_rev, placed_instance.size_rev, final_rot)
                                for cell in cells_to_remove:
                                    if cell in self.occupied_cells and self.occupied_cells[cell] == placed_instance.rail_index:
                                        del self.occupied_cells[cell]
                                
                                # 移除 OpenList 中该分叉的出口
                                num_exits = len(self.config_map[cand_id].exits_logic)
                                for _ in range(num_exits):
                                    self.open_list.pop()
                                
                                success = False
                                rail_success = False
                                # 继续尝试下一个候选
                        
                        else:
                            # 正常成功
                            success = True
                            rail_success = True
                            self.segment_diff_acc += placed_instance.diff_act
                            break
                    else:
                        fail_msg = result if result else "Unknown"
                        fail_reasons[fail_msg] = fail_reasons.get(fail_msg, 0) + 1
                
                if rail_success:
                    break
            
            if not success:
                print(f"Step Failed. Reasons: {fail_reasons}")
            
            dir_str = self.get_ue_dir_str(final_rot)
            if success:
                print(f"[Step {placed_instance.rail_index} Result] Exit_Pos_Rev={connector.target_pos.to_dict()}, Dir_Abs='{dir_str}', Attempts={attempts}, Success: Rail_ID={placed_id}, Rail_Index={placed_instance.rail_index}, Diff_Act={placed_instance.diff_act:.2f}, Backtracks={self.backtrack_count}")
            
            if must_end and success:
                print(f"已达到目标难度 ({self.current_total_difficulty}) 并放置终点。总回退次数: {self.backtrack_count}")
                break

    # 5.5 导出逻辑 (Export Logic)
    
    # 5.5.1 JSON 导出
    # 导出生成的迷宫数据到 JSON 文件。
    # 包含坐标转换、边界归一化和报告生成。
    def export_json(self, path):
        # 模式1: 动态边界后处理 - 归一化中心
        if BOUNDARY_MODE == 1 and self.placed_rails:
            print("Mode 1: Normalizing Maze Position...")
            
            # 使用 occupied_cells_rev 计算精确边界，避免隐式逻辑重复
            all_cells = [cell for r in self.placed_rails for cell in r.occupied_cells_rev]
            
            if not all_cells:
                 # Fallback if no cells (should not happen)
                 min_x, max_x = 0, 0
                 min_y, max_y = 0, 0
                 min_z, max_z = 0, 0
            else:
                min_x = min(c.x for c in all_cells)
                max_x = max(c.x for c in all_cells)
                
                min_y = min(c.y for c in all_cells)
                max_y = max(c.y for c in all_cells)
                
                min_z = min(c.z for c in all_cells)
                max_z = max(c.z for c in all_cells)
            
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            center_z = (min_z + max_z) / 2.0
            
            offset_x = -int(round(center_x))
            offset_y = -int(round(center_y))
            offset_z = -int(round(center_z))
            
            print(f"  -> Bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
            print(f"  -> Applying Offset: ({offset_x}, {offset_y}, {offset_z})")
            
            for r in self.placed_rails:
                r.pos_rev.x += offset_x
                r.pos_rev.y += offset_y
                r.pos_rev.z += offset_z
                
                for cell in r.occupied_cells_rev:
                    cell.x += offset_x
                    cell.y += offset_y
                    cell.z += offset_z

        # 生成 Markdown 报告内容
        report_lines = []
        report_lines.append("# Maze Generation Report")
        report_lines.append(f"Target Diff: {TARGET_DIFFICULTY}, Mode: {BOUNDARY_MODE}")
        report_lines.append("## Steps")
        report_lines.append("| Step | Rail ID | Rev Pos (X,Y,Z) | Size | Rot | Occupied Cells (Rev) |")
        report_lines.append("|---|---|---|---|---|---|")

        total_diff = 0.0
        json_rails = []
        
        for r in self.placed_rails:
            total_diff += r.diff_act
            
            occupied = [c.as_tuple() for c in r.occupied_cells_rev]
            occupied.sort()
            occ_str = "<br>".join([str(c) for c in occupied])
            
            report_lines.append(f"| {r.rail_index} | {r.rail_id} | {r.pos_rev.as_tuple()} | {r.size_rev.as_tuple()} | {r.rot_index} | {occ_str} |")
            
            # 计算物理坐标
            pos_abs = r.pos_rev.to_world_dict()
            pos_rev_dict = r.pos_rev.to_dict()
            
            # 物理旋转 (UE Rotator: P, Y, R)
            rot_abs = r.rot_abs
            dir_abs = self.get_ue_dir_str(r.rot_index)
            
            # 处理 Exits
            baked_exits = []
            cfg = self.config_map[r.rail_id]
            
            for i, status in enumerate(r.exit_status):
                logic_offset = cfg.exits_logic[i]['Pos']
                
                exit_local_rot_idx = cfg.exits_logic[i]['RotOffset']
                exit_abs_rot_idx = (r.rot_index + exit_local_rot_idx) % 4
                
                rotated_offset = logic_offset.rotate_z(r.rot_index)
                world_logic_pos = r.pos_rev.add(rotated_offset)
                world_phys_pos = world_logic_pos.to_world_dict()
                
                # 计算出口绝对旋转
                local_rot = cfg.exits_logic[i]['LocalRot']
                
                # Get Parent Abs Rot
                parent_p = r.rot_abs.get('p', 0.0)
                parent_y = r.rot_abs.get('y', 0.0)
                parent_r = r.rot_abs.get('r', 0.0)
                
                # Add components
                final_p = (parent_p + local_rot['p']) % 360.0
                final_y = (parent_y + local_rot['y']) % 360.0
                final_r = (parent_r + local_rot['r']) % 360.0
                
                exit_rot_abs = {
                    "p": final_p,
                    "y": final_y,
                    "r": final_r
                }
                
                # 计算绝对方向
                p = local_rot['p']
                if abs(p - 90.0) < 1.0:
                    exit_dir_abs = "+Z" # Up
                elif abs(p + 90.0) < 1.0 or abs(p - 270.0) < 1.0:
                    exit_dir_abs = "-Z" # Down
                else:
                    exit_dir_abs = self.get_ue_dir_str(exit_abs_rot_idx)
                
                baked_exits.append({
                    "Index": i,
                    "Exit_Pos_Rev": world_logic_pos.to_dict(),
                    "Exit_Pos_Abs": world_phys_pos,
                    "Exit_Rot_Abs": exit_rot_abs,
                    "Exit_Dir_Abs": exit_dir_abs,
                    "IsConnected": status['IsConnected'],
                    "TargetInstanceID": status['TargetID'] if status['TargetID'] != -1 else -1
                })

            json_rails.append({
                "Rail_Index": r.rail_index,
                "Rail_ID": r.rail_id,
                "Pos_Rev": pos_rev_dict,
                "Pos_Abs": pos_abs,
                "Rot_Abs": rot_abs,
                "Dir_Abs": dir_abs,
                "Size_Rev": r.size_rev.to_dict(),
                "Occupied_Cells_Rev": [c.to_dict() for c in r.occupied_cells_rev],
                "Diff_Base": 0,
                "Diff_Act": r.diff_act,
                "Prev_Index": r.prev_index,
                "Next_Index": r.next_indices,
                "Exit": baked_exits
            })

        out_data = {
            "MapMeta": {
                "LevelName": "Python_Generated_V2",
                "RailCount": len(json_rails),
                "MazeDiff": total_diff
            },
            "Rail": json_rails
        }
        with open(path, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Exported to {path}")

        # 写入 Markdown 报告
        with open("maze_generation_report.md", "w") as f:
            f.write("\n".join(report_lines))
        print("Exported Report to maze_generation_report.md")

# ==========================================
# 6. 程序入口 (Main Entry)
# ==========================================

# 6.1 主执行块
if __name__ == "__main__":
    try:
        # 加载配置
        configs = load_config('rail_config.csv')
        # 初始化生成器
        gen = MazeGenerator(configs)
        # 执行生成
        gen.generate()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        if 'gen' in locals():
            gen.export_json('maze_layout.json')
