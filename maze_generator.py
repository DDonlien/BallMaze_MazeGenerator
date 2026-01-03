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
        # 顺时针旋转
        for _ in range(rot_index % 4):
            # (x, y) -> (y, -x) 
            rx, ry = ry, -rx
        return Vector3(rx, ry, self.z)

# 修正 1.3: 迷宫最大尺寸 (逻辑单位)
MAZE_BOUNDS = Vector3(50, 50, 10) # X, Y, Z

# 修正 4: 目标难度
TARGET_DIFFICULTY = 100.0

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

@dataclass
class OpenConnector:
    # 修正 3.5: OpenList 存储的是“目标位置”和“目标旋转”
    target_pos: Vector3     # 下一个轨道应该放在哪个格子（逻辑）
    target_rot: int         # 下一个轨道应该保持什么旋转（逻辑索引 0-3）
    parent_id: int          # 父轨道 Index
    parent_exit_idx: int    # 父轨道的哪个出口
    accumulated_diff: float # 继承的难度
    diff_ratio: float       # 出口倍率

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
        name = row['Name']
        if pd.isna(name): continue
        
        diff = float(row['Difficulty']) if pd.notna(row['Difficulty']) else 0.0
        
        # Size
        sx = int(row['SizeX']) if pd.notna(row['SizeX']) else 1
        sy = int(row['SizeY']) if pd.notna(row['SizeY']) else 1
        sz = int(row['SizeZ']) if pd.notna(row['SizeZ']) else 1
        size = Vector3(sx, sy, sz)
        
        # Exits
        exits = []
        # Check Exit1, Exit2, Exit3...
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
        rail_type = str(row['Type']).lower() if pd.notna(row['Type']) else "normal"
        is_end = "end" in rail_type
        is_start = "start" in rail_type
        
        config_map[name] = RailConfigItem(name, diff, size, exits, is_end, is_start)
        
    return config_map

# ==========================================
# 4. 生成器核心
# ==========================================

class MazeGenerator:
    def __init__(self, config_map: Dict[str, RailConfigItem]):
        self.config_map = config_map
        self.placed_rails: List[RailInstance] = []
        self.occupied_cells: Set[Tuple[int, int, int]] = set()
        self.open_list: List[OpenConnector] = []
        self.global_index_counter = 0
        self.current_total_difficulty = 0.0

    def is_in_bounds(self, pos: Vector3, size: Vector3):
        # 修正 3.2: 检查迷宫范围
        # 假设迷宫中心在 0,0，范围是 -Bound/2 到 Bound/2
        half_x = MAZE_BOUNDS.x // 2
        half_y = MAZE_BOUNDS.y // 2
        
        # 检查四个角是否越界
        if pos.x < -half_x or pos.x + size.x > half_x: return False
        if pos.y < -half_y or pos.y + size.y > half_y: return False
        if pos.z < 0 or pos.z + size.z > MAZE_BOUNDS.z: return False # 假设Z从0开始向上
        return True

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
                        return True, None
        return False, Vector3(actual_size_x, actual_size_y, actual_size_z)

    def mark_occupied(self, pos: Vector3, actual_size: Vector3):
        for x in range(pos.x, pos.x + actual_size.x):
            for y in range(pos.y, pos.y + actual_size.y):
                for z in range(pos.z, pos.z + actual_size.z):
                    self.occupied_cells.add((x, y, z))

    def place_rail(self, rail_id: str, connector: OpenConnector = None, is_start=False):
        cfg = self.config_map[rail_id]
        
        # 确定位置和旋转
        if is_start:
            # 修正 1.3: 起点随机范围放置
            half_x, half_y = MAZE_BOUNDS.x // 2, MAZE_BOUNDS.y // 2
            start_x = random.randint(-half_x + 2, half_x - 2)
            start_y = random.randint(-half_y + 2, half_y - 2)
            
            pos = Vector3(start_x, start_y, 0)
            rot = 0 # 默认朝向
            diff_base = cfg.diff_base # 起点使用 Config 定义的难度
            ratio = 1.0
            prev_idx = -1
        else:
            pos = connector.target_pos
            rot = connector.target_rot
            diff_base = connector.accumulated_diff
            ratio = connector.diff_ratio
            prev_idx = connector.parent_id

        # 检查占用和边界
        colliding, actual_size = self.is_colliding(pos, cfg.size_rev, rot)
        if colliding: return None
        if not self.is_in_bounds(pos, actual_size): return None

        # 计算难度
        # 修正 4: 难度公式应用
        current_diff = (1.0 + diff_base * 0.1) * cfg.diff_base * ratio
        
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
        self.mark_occupied(pos, actual_size)
        self.placed_rails.append(instance)
        self.current_total_difficulty += current_diff

        # 处理新出口 -> 加入 OpenList
        for i, exit_data in enumerate(cfg.exits_logic):
            # 获取当前角度下的难度系数
            # SpinDiff = [0度, 90度, 180度, 270度]
            # 这里的 rot 是 Rail 的 Instance Rotation
            # 我们简化逻辑：当前 Rail 转了 rot，出口也就转了 rot
            # 真正可用的连接方向是：(ExitLocalRot + RailRot) % 4
            # 这里简化：假设所有接口都可以接，只要 SpinDiff > 0
            
            # 实际上我们这里需要算：这个出口现在在哪个世界逻辑坐标？
            local_pos = exit_data['Pos']
            rotated_offset = local_pos.rotate_z(rot) # 逻辑旋转
            world_exit_pos = pos.add(rotated_offset) # 世界逻辑坐标
            
            # 下一个轨道应该放在哪？
            # 假设简单的“平铺连接”：下一个轨道的原点(0,0,0) 对齐到这个出口位置
            # 假设下一个轨道的旋转 保持一致 (或者根据 SpinDiff 允许的旋转添加多个 Connector)
            
            # 为了简化，我们遍历 SpinDiff 的 4 个方向，大于 0 的都加入 OpenList
            spin_diffs = exit_data['SpinDiff']
            for spin_rot in range(4):
                ratio = spin_diffs[spin_rot]
                if ratio > 0:
                    # 绝对旋转 = (RailRot + SpinRot) % 4
                    target_rot = (rot + spin_rot) % 4
                    
                    self.open_list.append(OpenConnector(
                        target_pos=world_exit_pos,
                        target_rot=target_rot,
                        parent_id=idx,
                        parent_exit_idx=i,
                        accumulated_diff=current_diff,
                        diff_ratio=ratio
                    ))
                    
        return instance

    def generate(self):
        print(f"Start Generating... Target Diff: {TARGET_DIFFICULTY}")
        
        # 1. 放置 Start
        start_candidates = [k for k, v in self.config_map.items() if v.is_start]
        if not start_candidates: raise Exception("No Start Rail defined!")
        
        start_id = random.choice(start_candidates) # 随机选一个起点
        print(f"Placing Start: {start_id}")
        self.place_rail(start_id, is_start=True)

        # 2. 主循环
        while True:
            # 修正 4: 难度判断终止
            must_end = self.current_total_difficulty >= TARGET_DIFFICULTY
            
            if not self.open_list:
                print("迷宫生成结束 (OpenList Empty)")
                break

            # 修正 2.2: 随机获取 (不 shuffle 整个列表)
            # 修正 2.3: 移除已使用的 opening
            connector_idx = random.randint(0, len(self.open_list) - 1)
            connector = self.open_list.pop(connector_idx)
            
            # 筛选候选轨道
            if must_end:
                candidates = [k for k, v in self.config_map.items() if v.is_end]
            else:
                candidates = [k for k, v in self.config_map.items() if not v.is_end and not v.is_start]

            success = False
            # 随机尝试
            while candidates:
                cand_id = random.choice(candidates)
                candidates.remove(cand_id) # 试过就移除
                
                new_rail = self.place_rail(cand_id, connector)
                if new_rail:
                    # 更新父级连接信息
                    parent = next(r for r in self.placed_rails if r.rail_index == connector.parent_id)
                    parent.next_indices.append(new_rail.rail_index)
                    parent.exit_status[connector.parent_exit_idx]['IsConnected'] = True
                    parent.exit_status[connector.parent_exit_idx]['TargetID'] = new_rail.rail_index
                    
                    success = True
                    break
            
            # 修正 2.4: 死路处理
            if not success:
                print(f"遇到死路 at {connector.target_pos.as_tuple()}")
                # 检查父轨道有没有其他出口？
                # 由于我们维护的是全局 OpenList，只要 OpenList 不空，循环就会继续尝试其他接口
                # 也就是自然而然地“回溯”到了父级的其他出口（如果父级还有出口在 list 里）
                pass
            
            if must_end and success:
                print(f"已达到目标难度 ({self.current_total_difficulty}) 并放置终点。")
                break

    def export_json(self, path):
        # 修正 3.4 & 4.2: 导出时才计算物理坐标
        json_rails = []
        for r in self.placed_rails:
            # 计算物理坐标
            pos_abs = r.pos_rev.to_world_dict()
            
            # 物理旋转 (UE Rotator: P, Y, R)
            # 假设 Rot Index 对应 Yaw (Z轴) 0, 90, 180, 270
            rot_abs = {"p": 0.0, "y": float(r.rot_index * 90.0), "r": 0.0}
            
            # 处理 Exits 的物理坐标
            baked_exits = []
            cfg = self.config_map[r.rail_id]
            
            for i, status in enumerate(r.exit_status):
                # 重新计算出口物理位置用于连线
                logic_offset = cfg.exits_logic[i]['Pos']
                rotated_offset = logic_offset.rotate_z(r.rot_index)
                world_logic_pos = r.pos_rev.add(rotated_offset)
                world_phys_pos = world_logic_pos.to_world_dict()
                
                baked_exits.append({
                    "Index": i,
                    "Exit_Pos_Abs": world_phys_pos,
                    "IsConnected": status['IsConnected'],
                    "TargetInstanceID": status['TargetID'] if status['TargetID'] != -1 else -1
                })

            json_rails.append({
                "Rail_Index": r.rail_index,
                "Rail_ID": r.rail_id,
                "Pos_Abs": pos_abs,
                "Rot_Abs": rot_abs,
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
                "RailCount": len(json_rails)
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
        gen.export_json('maze_layout.json')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")