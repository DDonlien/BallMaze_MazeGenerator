import json
import random
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

# ==========================================
# 1. 核心配置与常量
# ==========================================

# 修正 3.1: 比例关系 1 BLOCK : 16 CM
GRID_TO_WORLD_SCALE = 16.0 

@dataclass
class Vector3:
    x: int = 0
    y: int = 0
    z: int = 0

    def to_dict(self): return {"x": self.x, "y": self.y, "z": self.z}
    def to_world_dict(self): return {"x": float(self.x * GRID_TO_WORLD_SCALE), "y": float(self.y * GRID_TO_WORLD_SCALE), "z": float(self.z * GRID_TO_WORLD_SCALE)}
    def as_tuple(self): return (self.x, self.y, self.z)
    
    # 向量加法
    def add(self, other): return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    # 简单的旋转逻辑 (仅处理 Z 轴 0, 90, 180, 270)
    # rot_index: 0=0, 1=90, 2=180, 3=270
    def rotate_z(self, rot_index):
        rx, ry = self.x, self.y
        # 顺时针旋转 -> 修正为逆时针/UE坐标系旋转 (0->X, 1->Y)
        for _ in range(rot_index % 4):
            # (x, y) -> (-y, x)
            rx, ry = -ry, rx
        return Vector3(rx, ry, self.z)

# 修正 1.3: 迷宫最大尺寸 (逻辑单位)
MAZE_BOUNDS = Vector3(4, 4, 1) # X, Y, Z (扩大边界以避免早期死路)

# 修正 4: 目标难度
TARGET_DIFFICULTY = 15.0
# 目标 Checkpoint 数量
TARGET_CHECKPOINTS = 0

# ==========================================
# 边界计算模式选项
# ==========================================
# 模式0: 静态边界（默认）。迷宫必须在 (-MAZE_BOUNDS, +MAZE_BOUNDS) 的绝对坐标范围内生成。
# 模式1: 动态边界。迷宫的“尺寸”不能超过 (MAZE_BOUNDS * 2)，但绝对位置可以浮动。
#       相当于迷宫可以在无限空间中生长，只要其 bounding box 不超过设定的大小。
#       生成完成后，会自动将迷宫移动到原点中心。
BOUNDARY_MODE = 1

# ==========================================
# 2. 数据结构
# ==========================================

@dataclass
class RailConfigItem:
    row_name: str
    diff_base: float
    size_rev: Vector3
    # 存储相对逻辑出口: List of (Pos_Rev, Rot_Index_Offset, SpinDiff)
    exits_logic: List[dict] 
    is_end: bool = False
    is_start: bool = False
    is_checkpoint: bool = False

@dataclass
class OpenConnector:
    target_pos: Vector3     # 下一个轨道应该放在哪个格子（逻辑）
    parent_id: int          # 父轨道 Index
    parent_exit_idx: int    # 父轨道的哪个出口
    accumulated_diff: float # 继承的难度
    
    # 新增: 存储父级旋转和 SpinDiff，用于在尝试时动态计算 target_rot
    parent_rot_index: int
    spin_diffs: List[float]
    parent_exit_rot_offset: int = 0 # 父级出口的固有旋转偏移 (0-3)
    forbidden_candidates: Set[str] = field(default_factory=set) # 避免回退后重复尝试同一死路

@dataclass
class RailInstance:
    rail_index: int
    rail_id: str
    pos_rev: Vector3    # 逻辑坐标
    rot_index: int      # 逻辑旋转 (0-3)
    size_rev: Vector3   # 逻辑尺寸
    diff_act: float     # 实际计算难度
    prev_index: int
    next_indices: List[int] = field(default_factory=list)
    # 存储出口连接状态 (用于 JSON 导出)
    exit_status: List[dict] = field(default_factory=list) 
    forbidden_siblings: Set[str] = field(default_factory=set) # 记录该节点生成时被禁用的兄弟节点 

# ==========================================
# 3. 解析与配置加载
# ==========================================

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
            
        # Size 解析
        # 新版: 从 Name 中解析 _X1_Y1_Z1
        # 旧版: 从 SizeX, SizeY, SizeZ 列读取
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
        
        # Exits 解析
        exits = []
        
        # 新版: Exit_Array 字符串解析
        if 'Exit_Array' in row and pd.notna(row['Exit_Array']):
            exit_str = str(row['Exit_Array'])
            # Pattern: Pos=(X=16,Y=0,Z=0),BaseRot=(P=0,Y=0,R=0)
            pat_pos = r"Pos=\(X=([\d.-]+),Y=([\d.-]+),Z=([\d.-]+)\)"
            pat_rot = r"BaseRot=\(P=([\d.-]+),Y=([\d.-]+),R=([\d.-]+)\)"
            
            pos_matches = list(re.finditer(pat_pos, exit_str))
            rot_matches = list(re.finditer(pat_rot, exit_str))
            
            for i, p_match in enumerate(pos_matches):
                px, py, pz = map(float, p_match.groups())
                
                # Convert world coords to grid coords
                # round to nearest integer to be safe against float errors
                gx = int(round(px / GRID_TO_WORLD_SCALE))
                gy = int(round(py / GRID_TO_WORLD_SCALE))
                gz = int(round(pz / GRID_TO_WORLD_SCALE))
                logic_pos = Vector3(gx, gy, gz)
                
                # 解析 BaseRot
                rot = 0
                if i < len(rot_matches):
                    _, ry, _ = map(float, rot_matches[i].groups())
                    rot_deg = int(ry)
                    # Normalize to 0-3 index
                    # 90 -> 1, 180 -> 2, -90 -> 3 (270)
                    rot = int(rot_deg // 90) % 4
                
                # 解析 SpinDiff
                
                exits.append({
                    "Pos": logic_pos,
                    "RotOffset": rot,
                    "SpinDiff": [1.0, 1.0, 1.0, 1.0]
                })

        # 旧版: Exit1Pos, Exit1Rot...
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
                        
                        exits.append({
                            "Pos": logic_pos,
                            "RotOffset": rot,
                            # Default SpinDiff to allow all rotations with weight 1.0
                            "SpinDiff": [1.0, 1.0, 1.0, 1.0] 
                        })
                    except ValueError:
                        pass
        
        # Type
        rail_type = "normal"
        if 'Type' in row and pd.notna(row['Type']):
            rail_type = str(row['Type']).lower()
        else:
            if "start" in name.lower(): rail_type = "start"
            elif "end" in name.lower(): rail_type = "end"
            
        is_end = "end" in rail_type
        is_start = "start" in rail_type
        is_checkpoint = "checkpoint" in rail_type or "checkpoint" in name.lower()
        
        config_map[name] = RailConfigItem(name, diff, size, exits, is_end, is_start, is_checkpoint)
        
    return config_map

# ==========================================
# 4. 生成器核心
# ==========================================

class MazeGenerator:
    def __init__(self, config_map: Dict[str, RailConfigItem]):
        self.config_map = config_map
        self.placed_rails: List[RailInstance] = []
        self.occupied_cells: Dict[Tuple[int, int, int], int] = {}
        self.open_list: List[OpenConnector] = []
        self.global_index_counter = 0
        self.current_total_difficulty = 0.0
        
        # Checkpoint State
        self.target_checkpoints = TARGET_CHECKPOINTS # 默认值，实际可由外部设置
        self.placed_checkpoints_count = 0
        self.segment_diff_acc = 0.0
        
        self.backtrack_count = 0

    def is_in_bounds(self, pos: Vector3, size: Vector3):
        # 修正 3.2: 检查迷宫范围
        
        if BOUNDARY_MODE == 0:
            # 模式0: 静态边界检查
            # 范围是 [-Bound, Bound] (inclusive)
            if pos.x < -MAZE_BOUNDS.x or pos.x + size.x - 1 > MAZE_BOUNDS.x: return False
            if pos.y < -MAZE_BOUNDS.y or pos.y + size.y - 1 > MAZE_BOUNDS.y: return False
            if pos.z < -MAZE_BOUNDS.z or pos.z + size.z - 1 > MAZE_BOUNDS.z: return False 
            return True
            
        elif BOUNDARY_MODE == 1:
            # 模式1: 动态边界检查 (检查 Bounding Box 大小)
            # 计算加入当前新 Rail 后的整体 Min/Max
            
            # 1. 当前已有的 Bounding Box
            if not self.placed_rails:
                # 只有当前这一个，肯定在范围内（假设单个 Rail 不会超过 Bounds * 2）
                # 但为了严谨，还是检查一下 size 是否超过 2*Bounds
                # Bound 是半长，所以全长是 Bound*2 + 1 (如果含0) 或者 Bound*2
                # 这里我们沿用 Mode 0 的逻辑: -Bound 到 +Bound，跨度是 2*Bound + 1
                
                max_span_x = MAZE_BOUNDS.x * 2 + 1
                max_span_y = MAZE_BOUNDS.y * 2 + 1
                max_span_z = MAZE_BOUNDS.z * 2 + 1
                
                if size.x > max_span_x or size.y > max_span_y or size.z > max_span_z:
                    return False
                return True

            # 获取当前所有 Rail 的边界 (这步可能需要优化，如果太慢可以缓存)
            # 这里的 pos_rev 是左下角
            curr_min_x = min(r.pos_rev.x for r in self.placed_rails)
            curr_max_x = max(r.pos_rev.x + (r.size_rev.x if r.rot_index % 2 == 0 else r.size_rev.y) - 1 for r in self.placed_rails)
            
            curr_min_y = min(r.pos_rev.y for r in self.placed_rails)
            curr_max_y = max(r.pos_rev.y + (r.size_rev.y if r.rot_index % 2 == 0 else r.size_rev.x) - 1 for r in self.placed_rails)
            
            curr_min_z = min(r.pos_rev.z for r in self.placed_rails)
            curr_max_z = max(r.pos_rev.z + r.size_rev.z - 1 for r in self.placed_rails)
            
            # 2. 加入新 Rail 后的边界
            new_min_x = min(curr_min_x, pos.x)
            new_max_x = max(curr_max_x, pos.x + size.x - 1)
            
            new_min_y = min(curr_min_y, pos.y)
            new_max_y = max(curr_max_y, pos.y + size.y - 1)
            
            new_min_z = min(curr_min_z, pos.z)
            new_max_z = max(curr_max_z, pos.z + size.z - 1)
            
            # 3. 检查尺寸是否超过允许范围
            # 允许范围是 Mode 0 的跨度: [-B, B] -> span = 2*B + 1
            allowed_span_x = MAZE_BOUNDS.x * 2 + 1
            allowed_span_y = MAZE_BOUNDS.y * 2 + 1
            allowed_span_z = MAZE_BOUNDS.z * 2 + 1
            
            if (new_max_x - new_min_x + 1) > allowed_span_x: return False
            if (new_max_y - new_min_y + 1) > allowed_span_y: return False
            if (new_max_z - new_min_z + 1) > allowed_span_z: return False
            
            return True
            
        return False

    def is_colliding(self, pos: Vector3, size: Vector3, rot_idx: int):
        # 计算旋转后的实际占地尺寸
        # 如果旋转 90 (1) 或 270 (3)，X 和 Y 尺寸互换
        actual_size_x = size.x if rot_idx % 2 == 0 else size.y
        actual_size_y = size.y if rot_idx % 2 == 0 else size.x
        actual_size_z = size.z

        for x in range(pos.x, pos.x + actual_size_x):
            for y in range(pos.y, pos.y + actual_size_y):
                for z in range(pos.z, pos.z + actual_size_z):
                    if (x, y, z) in self.occupied_cells:
                        return True, self.occupied_cells[(x, y, z)]
        return False, Vector3(actual_size_x, actual_size_y, actual_size_z)

    def mark_occupied(self, pos: Vector3, actual_size: Vector3, rail_index: int):
        for x in range(pos.x, pos.x + actual_size.x):
            for y in range(pos.y, pos.y + actual_size.y):
                for z in range(pos.z, pos.z + actual_size.z):
                    self.occupied_cells[(x, y, z)] = rail_index

    def place_rail(self, rail_id: str, connector: OpenConnector = None, is_start=False):
        cfg = self.config_map[rail_id]
        
        # 确定位置和旋转
        if is_start:
            # 修正 1.3: 起点随机范围放置
            half_x, half_y = MAZE_BOUNDS.x // 2, MAZE_BOUNDS.y // 2
            # 确保范围有效
            range_x = max(1, half_x - 5)
            range_y = max(1, half_y - 5)
            start_x = random.randint(-range_x, range_x)
            start_y = random.randint(-range_y, range_y)
            
            pos = Vector3(start_x, start_y, 0)
            rot = 0 # 默认朝向
            diff_base = cfg.diff_base # 起点使用 Config 定义的难度
            ratio = 1.0
            prev_idx = -1
            print(f"-> 尝试放置 Start [{rail_id}] at Pos={pos.as_tuple()}")
        else:
            # 这里需要外部循环传入确定的 rot 和 ratio
            # 为了保持 place_rail 接口简单，我们假设 connector 已经被拆解或者外部传入参数
            # 但这里我们调整 place_rail 接口会比较大，
            # 我们可以保留 place_rail 做核心检测，参数改一下
            pass
            
        # 这里的重构逻辑：
        # place_rail 不应该再负责从 connector 拆解 rot，而是应该接受明确的 pos, rot
        # 所以我们需要修改 place_rail 的签名，或者在内部处理
        
        # 临时方案：is_start 特殊处理，非 start 时，我们期望调用者已经算好了 pos, rot
        # 但原来的调用是 self.place_rail(cand_id, connector)
        # 我们改成 self.place_rail(cand_id, pos, rot, diff_base, ratio, prev_idx)
        pass

    def place_rail_v2(self, rail_id: str, pos: Vector3, rot: int, diff_base_acc: float, ratio: float, prev_idx: int):
        cfg = self.config_map[rail_id]
        
        # 检查占用和边界
        colliding, result_data = self.is_colliding(pos, cfg.size_rev, rot)
        if colliding: 
            return f"Collision with Rail {result_data}"
            
        if not self.is_in_bounds(pos, result_data): 
            return "OutOfBounds"
        
        actual_size = result_data

        # 计算难度
        # 修正 4: 难度公式应用
        current_diff = (1.0 + diff_base_acc * 0.1) * cfg.diff_base * ratio
        
        # 实例化
        idx = self.global_index_counter
        self.global_index_counter += 1
        
        instance = RailInstance(
            rail_index=idx,
            rail_id=rail_id,
            pos_rev=pos,
            rot_index=rot,
            size_rev=cfg.size_rev, # 存原始逻辑尺寸
            diff_act=current_diff,
            prev_index=prev_idx
        )
        
        # 初始化出口状态
        instance.exit_status = [{"Index": i, "IsConnected": False, "TargetID": -1, "WorldPos": None} for i in range(len(cfg.exits_logic))]

        # 更新全局
        self.mark_occupied(pos, actual_size, idx)
        self.placed_rails.append(instance)
        self.current_total_difficulty += current_diff

        # 处理新出口 -> 加入 OpenList
        for i, exit_data in enumerate(cfg.exits_logic):
            local_pos = exit_data['Pos']
            rotated_offset = local_pos.rotate_z(rot) # 逻辑旋转
            world_exit_pos = pos.add(rotated_offset) # 世界逻辑坐标
            
            spin_diffs = exit_data['SpinDiff']
            exit_rot_offset = exit_data.get('RotOffset', 0)
            
            # 加入 OpenList，不再拆解 spin
            self.open_list.append(OpenConnector(
                target_pos=world_exit_pos,
                parent_id=idx,
                parent_exit_idx=i,
                accumulated_diff=current_diff,
                parent_rot_index=rot,
                spin_diffs=spin_diffs,
                parent_exit_rot_offset=exit_rot_offset
            ))
                    
        return instance

    def get_ue_dir_str(self, rot_idx):
        # 0=0(+X), 1=90(+Y), 2=180(-X), 3=270(-Y)
        dirs = ["+X", "+Y", "-X", "-Y"]
        return dirs[rot_idx % 4]

    def generate(self):
        print(f"Start Generating... Target Diff: {TARGET_DIFFICULTY}")
        
        # 1. 放置 Start
        start_candidates = [k for k, v in self.config_map.items() if v.is_start]
        if not start_candidates: raise Exception("No Start Rail defined!")
        
        start_id = random.choice(start_candidates) # 随机选一个起点
        print(f"Placing Start: {start_id}")
        
        # 修正 1.3: 起点随机范围放置 (Radius Mode)
        # 获取 Start Rail 尺寸
        start_cfg = self.config_map[start_id]
        start_sz = start_cfg.size_rev
        
        # 计算合法范围 [-Bound, Bound - Size + 1]
        min_x, max_x = -MAZE_BOUNDS.x, MAZE_BOUNDS.x - start_sz.x + 1
        min_y, max_y = -MAZE_BOUNDS.y, MAZE_BOUNDS.y - start_sz.y + 1
        # min_z, max_z = -MAZE_BOUNDS.z, MAZE_BOUNDS.z - start_sz.z + 1 

        # 确保 min <= max
        if min_x > max_x: min_x = max_x = -MAZE_BOUNDS.x 
        if min_y > max_y: min_y = max_y = -MAZE_BOUNDS.y

        start_x = random.randint(min_x, max_x)
        start_y = random.randint(min_y, max_y)
        
        start_pos = Vector3(start_x, start_y, 0)
        start_rot = 0
        print(f"-> 尝试放置 Start [{start_id}] at Pos={start_pos.as_tuple()}, Dir_Abs={self.get_ue_dir_str(start_rot)}")
        
        self.place_rail_v2(start_id, start_pos, start_rot, 0, 1.0, -1)
        
        # Calculate Segment Difficulty
        segment_target_diff = TARGET_DIFFICULTY / (self.target_checkpoints + 1)
        print(f"Segment Target Diff: {segment_target_diff}")

        # 2. 主循环
        while True:
            # 修正 4: 难度判断终止
            must_end = self.current_total_difficulty >= TARGET_DIFFICULTY
            
            # Checkpoint Trigger
            trigger_checkpoint = (self.placed_checkpoints_count < self.target_checkpoints) and \
                                 (self.segment_diff_acc >= segment_target_diff)
            
            if not self.open_list:
                # print("当前路径无解 (OpenList Empty)，尝试回退...")
                
                # Backtracking Logic for Dead End (when OpenList is empty)
                if self.placed_rails:
                    # Implement Backtracking:
                    self.backtrack_count += 1
                    # 1. Pop last rail
                    last_rail = self.placed_rails.pop()
                    self.global_index_counter -= 1 # 修正索引计数，避免无限增长
                    self.current_total_difficulty -= last_rail.diff_act
                    
                    # 2. Unmark occupied cells
                    # Re-calc size for unmark
                    sz = last_rail.size_rev
                    rot_idx = last_rail.rot_index
                    actual_size_x = sz.x if rot_idx % 2 == 0 else sz.y
                    actual_size_y = sz.y if rot_idx % 2 == 0 else sz.x
                    actual_size_z = sz.z
                    
                    pos = last_rail.pos_rev
                    for x in range(pos.x, pos.x + actual_size_x):
                        for y in range(pos.y, pos.y + actual_size_y):
                            for z in range(pos.z, pos.z + actual_size_z):
                                if (x, y, z) in self.occupied_cells and self.occupied_cells[(x,y,z)] == last_rail.rail_index:
                                    del self.occupied_cells[(x, y, z)]
                    
                    # 3. Recover Parent Connector
                    # We need to find the parent and re-activate the exit
                    if last_rail.prev_index != -1:
                        parent = next((r for r in self.placed_rails if r.rail_index == last_rail.prev_index), None)
                        if parent:
                            # Find which exit connected to last_rail
                            # Since last_rail doesn't store which parent exit it came from explicitly (OpenConnector did),
                            # we have to search parent's exit_status for TargetID == last_rail.rail_index
                            for exit_idx, status in enumerate(parent.exit_status):
                                if status['TargetID'] == last_rail.rail_index:
                                    status['IsConnected'] = False
                                    status['TargetID'] = -1
                                    
                                    # Re-create OpenConnector
                                    # We need original logic pos and spin diffs
                                    # We can re-fetch from config
                                    parent_cfg = self.config_map[parent.rail_id]
                                    exit_data = parent_cfg.exits_logic[exit_idx]
                                    
                                    # Recalculate world pos
                                    local_pos = exit_data['Pos']
                                    rotated_offset = local_pos.rotate_z(parent.rot_index)
                                    world_exit_pos = parent.pos_rev.add(rotated_offset)
                                    
                                    spin_diffs = exit_data['SpinDiff']
                                    exit_rot_offset = exit_data.get('RotOffset', 0)
                                    
                                    # 继承之前的 forbidden siblings，并加上当前失败的 rail_id
                                    new_forbidden = last_rail.forbidden_siblings.copy()
                                    new_forbidden.add(last_rail.rail_id)

                                    # Add back to OpenList
                                    self.open_list.append(OpenConnector(
                                        target_pos=world_exit_pos,
                                        parent_id=parent.rail_index,
                                        parent_exit_idx=exit_idx,
                                        accumulated_diff=self.current_total_difficulty, # Approx
                                        parent_rot_index=parent.rot_index,
                                        spin_diffs=spin_diffs,
                                        parent_exit_rot_offset=exit_rot_offset,
                                        forbidden_candidates=new_forbidden
                                    ))
                                    # print(f"Backtracked to Rail {parent.rail_index}, Exit {exit_idx}. Forbidden now: {new_forbidden}")
                                    break
                    continue # Retry loop
                
                break

            # 修正 2.2: 随机获取 (不 shuffle 整个列表)
            # 修正 2.3: 移除已使用的 opening
            connector_idx = random.randint(0, len(self.open_list) - 1)
            connector = self.open_list.pop(connector_idx)
            
            # 筛选候选轨道
            if must_end:
                candidates = [k for k, v in self.config_map.items() if v.is_end]
            elif trigger_checkpoint:
                # Checkpoint Phase 1: Must place a Fork (Rail with >= 2 Exits)
                # Filter candidates: Not End, Not Start, Not Checkpoint, Exits >= 2
                candidates = [k for k, v in self.config_map.items() 
                              if not v.is_end and not v.is_start and not v.is_checkpoint and len(v.exits_logic) >= 2]
                if not candidates:
                    print("Warning: No Fork Rails available for Checkpoint placement! Skipping Checkpoint logic this step.")
                    candidates = [k for k, v in self.config_map.items() if not v.is_end and not v.is_start and not v.is_checkpoint]
            else:
                candidates = [k for k, v in self.config_map.items() if not v.is_end and not v.is_start and not v.is_checkpoint]
            
            # 过滤掉 Forbidden Candidates (来自回退逻辑)
            original_candidate_count = len(candidates)
            candidates = [c for c in candidates if c not in connector.forbidden_candidates]
            if original_candidate_count > 0 and not candidates:
                # 如果过滤后没候选了，说明这个 Connector 彻底废了 (Dead End)
                # 我们不需要显式做任何事，因为 candidates 为空，下面的 while 循环不执行
                # success=False，fail_reasons 空 (或 dummy)
                # 然后进入下一轮循环。如果 OpenList 空了，就会触发 Backtracking。
                # 这样就实现了 DFS 式的深层回退。
                print(f"All candidates forbidden for this connector (Tried: {connector.forbidden_candidates}). Skipping.")
                pass

            success = False
            fail_reasons = {}
            attempts = 0
            placed_id = None
            placed_instance = None
            final_rot = 0
            
            # 计算所有可能的旋转
            # (target_rot, ratio)
            possible_rots = []
            for spin_rot, ratio in enumerate(connector.spin_diffs):
                if ratio > 0:
                    # 绝对旋转 = (ParentRot + ExitRotOffset + SpinRot) % 4
                    # 修正: 必须包含 ExitRotOffset，否则转向逻辑丢失
                    target_rot = (connector.parent_rot_index + connector.parent_exit_rot_offset + spin_rot) % 4
                    possible_rots.append((target_rot, ratio))

            # 随机尝试
            while candidates:
                cand_id = random.choice(candidates)
                candidates.remove(cand_id) # 试过就移除
                
                # 在这个 Rail 尝试所有可能的旋转
                rail_success = False
                
                for rot, ratio in possible_rots:
                    attempts += 1
                    
                    # 尝试放置
                    result = self.place_rail_v2(cand_id, connector.target_pos, rot, connector.accumulated_diff, ratio, connector.parent_id)
                    
                    if isinstance(result, RailInstance):
                        # 更新父级连接信息
                        parent = next(r for r in self.placed_rails if r.rail_index == connector.parent_id)
                        parent.next_indices.append(result.rail_index)
                        parent.exit_status[connector.parent_exit_idx]['IsConnected'] = True
                        parent.exit_status[connector.parent_exit_idx]['TargetID'] = result.rail_index
                        
                        placed_id = cand_id
                        placed_instance = result
                        placed_instance.forbidden_siblings = connector.forbidden_candidates.copy() # 记录父级 Connector 的禁忌表
                        final_rot = rot
                        
                        # Checkpoint Logic Phase 2: If we are placing a Fork for checkpoint, try to place checkpoint now
                        if trigger_checkpoint:
                            # Find a valid exit for checkpoint
                            checkpoint_placed_success = False
                            checkpoint_candidates = [k for k, v in self.config_map.items() if v.is_checkpoint]
                            
                            if not checkpoint_candidates:
                                print("Error: No Checkpoint Rails defined but Trigger is active!")
                                # Fallback: Treat as normal success, skip checkpoint
                                success = True
                                rail_success = True
                                break

                            # Try to place Checkpoint on one of the exits
                            # We iterate through the exits of the newly placed Fork
                            for exit_idx, status in enumerate(placed_instance.exit_status):
                                # Skip if already connected (unlikely for new rail)
                                if status['IsConnected']: continue
                                
                                # Prepare parameters for Checkpoint placement
                                parent_cfg = self.config_map[placed_instance.rail_id]
                                exit_data = parent_cfg.exits_logic[exit_idx]
                                
                                local_pos = exit_data['Pos']
                                rotated_offset = local_pos.rotate_z(final_rot)
                                world_exit_pos = placed_instance.pos_rev.add(rotated_offset)
                                
                                spin_diffs = exit_data['SpinDiff']
                                exit_rot_offset = exit_data.get('RotOffset', 0)
                                parent_rot_idx = final_rot
                                
                                # Try each Checkpoint candidate
                                for cp_id in checkpoint_candidates:
                                    # Try rotations for Checkpoint
                                    # Since Checkpoint usually has 1 input, rotation matters less? 
                                    # Or we align with parent exit direction?
                                    # Let's assume standard rotation logic
                                    
                                    # Calculate target rot
                                    # We try all valid spins
                                    for spin_rot, spin_ratio in enumerate(spin_diffs):
                                        if spin_ratio <= 0: continue
                                        
                                        target_rot = (parent_rot_idx + exit_rot_offset + spin_rot) % 4
                                        
                                        # Try Place
                                        cp_result = self.place_rail_v2(cp_id, world_exit_pos, target_rot, 
                                                                     placed_instance.diff_act + connector.accumulated_diff, # Accumulate Diff?
                                                                     spin_ratio, placed_instance.rail_index)
                                        
                                        if isinstance(cp_result, RailInstance):
                                            # Success!
                                            print(f"  -> Checkpoint Placed: {cp_id} at Exit {exit_idx}")
                                            
                                            # Connect Fork -> Checkpoint
                                            placed_instance.next_indices.append(cp_result.rail_index)
                                            placed_instance.exit_status[exit_idx]['IsConnected'] = True
                                            placed_instance.exit_status[exit_idx]['TargetID'] = cp_result.rail_index
                                            
                                            # Update Global State
                                            self.placed_checkpoints_count += 1
                                            self.segment_diff_acc = 0.0 # Reset Segment Diff
                                            checkpoint_placed_success = True
                                            break # Spin Loop
                                    
                                    if checkpoint_placed_success: break # CP Cand Loop
                                
                                if checkpoint_placed_success: break # Exit Loop
                            
                            if checkpoint_placed_success:
                                success = True
                                rail_success = True
                                break # Rot Loop (Fork)
                            else:
                                # Failed to place Checkpoint on this Fork
                                # Rollback Fork
                                print(f"  -> Failed to place Checkpoint on Fork {cand_id}, Rolling back Fork.")
                                # Remove from placed_rails
                                self.placed_rails.pop() # Remove Fork
                                self.global_index_counter -= 1 # Restore ID
                                self.current_total_difficulty -= placed_instance.diff_act
                                
                                # Unmark Occupied
                                # Re-calc size for unmark
                                sz = placed_instance.size_rev
                                actual_sz_x = sz.x if final_rot % 2 == 0 else sz.y
                                actual_sz_y = sz.y if final_rot % 2 == 0 else sz.x
                                actual_sz_z = sz.z
                                for x in range(placed_instance.pos_rev.x, placed_instance.pos_rev.x + actual_sz_x):
                                    for y in range(placed_instance.pos_rev.y, placed_instance.pos_rev.y + actual_sz_y):
                                        for z in range(placed_instance.pos_rev.z, placed_instance.pos_rev.z + actual_sz_z):
                                            if (x,y,z) in self.occupied_cells and self.occupied_cells[(x,y,z)] == placed_instance.rail_index:
                                                del self.occupied_cells[(x,y,z)]
                                
                                # Remove Fork's Exits from OpenList
                                # They were added at the end of open_list
                                num_exits = len(self.config_map[cand_id].exits_logic)
                                for _ in range(num_exits):
                                    self.open_list.pop()
                                
                                success = False
                                rail_success = False
                                # Continue to next Fork Rotation/Candidate
                        
                        else:
                            # Normal Success
                            success = True
                            rail_success = True
                            self.segment_diff_acc += placed_instance.diff_act # Update Segment Diff
                            break # 跳出旋转循环
                    else:
                        fail_msg = result if result else "Unknown"
                        fail_reasons[fail_msg] = fail_reasons.get(fail_msg, 0) + 1
                
                if rail_success:
                    break # 跳出候选循环
            
            dir_str = self.get_ue_dir_str(final_rot)
            if success:
                print(f"[Step {placed_instance.rail_index} Result] Exit_Pos_Rev={connector.target_pos.to_dict()}, Dir_Abs='{dir_str}', Attempts={attempts}, Success: Rail_ID={placed_id}, Rail_Index={placed_instance.rail_index}, Diff_Act={placed_instance.diff_act:.2f}, Backtracks={self.backtrack_count}")
            else:
                # 失败时 Dir_Abs 可能有多个，这里只打印最后一次尝试的
                # print(f"[Step {self.global_index_counter} Result] Exit_Pos_Rev={connector.target_pos.to_dict()}, Attempts={attempts}, Failed: {fail_reasons}")
                pass
            
            if must_end and success:
                print(f"已达到目标难度 ({self.current_total_difficulty}) 并放置终点。总回退次数: {self.backtrack_count}")
                break

    def export_json(self, path):
        # 修正 3.4 & 4.2: 导出时才计算物理坐标
        
        # 模式1: 动态边界后处理 - 归一化中心
        if BOUNDARY_MODE == 1 and self.placed_rails:
            print("Mode 1: Normalizing Maze Position...")
            # 1. 计算最终包围盒
            min_x = min(r.pos_rev.x for r in self.placed_rails)
            max_x = max(r.pos_rev.x + (r.size_rev.x if r.rot_index % 2 == 0 else r.size_rev.y) - 1 for r in self.placed_rails)
            
            min_y = min(r.pos_rev.y for r in self.placed_rails)
            max_y = max(r.pos_rev.y + (r.size_rev.y if r.rot_index % 2 == 0 else r.size_rev.x) - 1 for r in self.placed_rails)
            
            # min_z = min(r.pos_rev.z for r in self.placed_rails)
            # max_z = max(r.pos_rev.z + r.size_rev.z - 1 for r in self.placed_rails)
            
            # 2. 计算中心偏差
            # 目标是将 (min + max) / 2 移动到 0
            # offset = 0 - center
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            # center_z = (min_z + max_z) / 2.0 
            
            # 取整偏移量 (保持 Grid 对齐)
            offset_x = -int(round(center_x))
            offset_y = -int(round(center_y))
            # offset_z = -int(round(center_z)) # Z 轴通常从 0 开始，或者也归一化？ 假设保持 Z 相对位置，或者也归一化。通常迷宫在地平面，Z可能不需要动，或者归一化到底部为0？
            # 用户的需求是“迷宫中央位置”，通常指 XY 平面。Z 轴如果归一化可能会导致埋地。暂只调整 XY。
            
            print(f"  -> Bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
            print(f"  -> Center: ({center_x}, {center_y})")
            print(f"  -> Applying Offset: ({offset_x}, {offset_y})")
            
            # 3. 应用偏移
            for r in self.placed_rails:
                r.pos_rev.x += offset_x
                r.pos_rev.y += offset_y
                # r.pos_rev.z += offset_z

        total_diff = 0.0
        json_rails = []
        for r in self.placed_rails:
            total_diff += r.diff_act
            
            # 计算物理坐标
            pos_abs = r.pos_rev.to_world_dict()
            pos_rev_dict = r.pos_rev.to_dict()
            
            # 物理旋转 (UE Rotator: P, Y, R)
            # 假设 Rot Index 对应 Yaw (Z轴) 0, 90, 180, 270
            rot_abs = {"p": 0.0, "y": float(r.rot_index * 90.0), "r": 0.0}
            dir_abs = self.get_ue_dir_str(r.rot_index)
            
            # 处理 Exits 的物理坐标
            baked_exits = []
            cfg = self.config_map[r.rail_id]
            
            for i, status in enumerate(r.exit_status):
                # 重新计算出口物理位置用于连线
                logic_offset = cfg.exits_logic[i]['Pos']
                
                # Exit 的绝对旋转索引 = (Rail旋转 + Exit相对旋转) % 4
                exit_local_rot = cfg.exits_logic[i]['RotOffset']
                exit_abs_rot_idx = (r.rot_index + exit_local_rot) % 4
                
                # 计算世界逻辑坐标
                rotated_offset = logic_offset.rotate_z(r.rot_index)
                world_logic_pos = r.pos_rev.add(rotated_offset)
                
                world_phys_pos = world_logic_pos.to_world_dict()
                
                # Exit 详细属性
                exit_rot_abs = {"p": 0.0, "y": float(exit_abs_rot_idx * 90.0), "r": 0.0}
                exit_dir_abs = self.get_ue_dir_str(exit_abs_rot_idx)
                
                baked_exits.append({
                    "Index": i,
                    "Exit_Pos_Rev": world_logic_pos.to_dict(), # Unified Prefix
                    "Exit_Pos_Abs": world_phys_pos,
                    "Exit_Rot_Abs": exit_rot_abs, # Unified Prefix
                    "Exit_Dir_Abs": exit_dir_abs, # Unified Prefix
                    "IsConnected": status['IsConnected'],
                    "TargetInstanceID": status['TargetID'] if status['TargetID'] != -1 else -1
                })

            json_rails.append({
                "Rail_Index": r.rail_index,
                "Rail_ID": r.rail_id,
                "Pos_Rev": pos_rev_dict, # 新增
                "Pos_Abs": pos_abs,
                "Rot_Abs": rot_abs,
                "Dir_Abs": dir_abs, # 新增
                "Size_Rev": r.size_rev.to_dict(),
                "Diff_Base": 0, # 这里略过，只看 Final
                "Diff_Act": r.diff_act,
                "Prev_Index": r.prev_index,
                "Next_Index": r.next_indices,
                "Exit": baked_exits
            })

        out_data = {
            "MapMeta": {
                "LevelName": "Python_Generated_V2",
                "RailCount": len(json_rails),
                "MazeDiff": total_diff # 新增
            },
            "Rail": json_rails
        }
        with open(path, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Exported to {path}")

# ==========================================
# 5. 执行
# ==========================================

if __name__ == "__main__":
    # 请确保文件路径正确
    try:
        # configs = load_config('迷宫球 - 配置表 - RailConfig.csv', '迷宫球 - 配置表 - RailDefinition.csv')
        configs = load_config('rail_config.csv')
        gen = MazeGenerator(configs)
        gen.generate()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        if 'gen' in locals():
            gen.export_json('maze_layout.json')
