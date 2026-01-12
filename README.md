# Fershli4 Maze Generator & Viewer

这是一个基于 Python 的程序化内容生成 (PCG) 工具套件，专为 Roguelike 游戏（如“跃入迷城”）设计。它负责读取 UE5 导出的配置表，生成逻辑严密的 3D 迷宫布局，并提供 H5 可视化工具以便快速检阅。

## ✨ 主要特性 (Features)

*   **程序化生成 (PCG)**: 基于“难度驱动的生长算法”，自动构建复杂的 3D 轨道迷宫。
*   **UE5 工作流兼容**: 直接读取 UE5 导出的 CSV 轨道配置，输出可直接导入 UE5 的 JSON 数据。
*   **严密的逻辑校验**: 
    *   **全整数逻辑坐标**: 杜绝浮点数误差带来的对齐问题。
    *   **AABB 碰撞检测**: 确保生成的轨道互不穿插。
    *   **连通性保证**: 自动回溯死路，确保路径有效。
*   **Web 可视化**: 内置 Three.js 开发的 H5 查看器，无需启动引擎即可快速预览迷宫结构和难度分布。

## 📁 项目结构 (Project Structure)

```text
.
├── maze_generator.py          # [核心] 迷宫生成脚本
├── render_maze.html           # [工具] H5 3D 迷宫查看器 (基于 Three.js)
├── rail_config.csv            # [配置] 轨道零件配置表 (UE导出)
├── template_maze_layout.json  # [参考] JSON 数据结构模板
├── README.md                  # [文档] 项目说明文档
└── output/                    # [输出] 生成结果目录
    └── maze_layout_xxxx.json  # 自动生成的迷宫文件
```

## 🛠️ 快速开始 (Getting Started)

### 环境需求

*   **Python**: 3.8 或更高版本
*   **依赖库**: `pandas`

### 安装

1.  克隆仓库或下载源码。
2.  安装必要的 Python 依赖：

```bash
pip install pandas
```

### 使用方法

#### 1. 生成迷宫

直接运行 Python 脚本。脚本会自动加载同目录下的 `rail_config.csv`。

```bash
python maze_generator.py
```

*   **输出**: 成功执行后，会在 `output/` 目录下生成带有时间戳的 JSON 文件，例如 `maze_layout_202401031530.json`。
*   **日志**: 控制台会打印生成过程，包括起点选择、生成进度、死路回溯以及最终的难度统计。

#### 2. 可视化检阅

1.  使用浏览器（推荐 Chrome 或 Edge）打开 `render_maze.html`。
2.  找到生成的 JSON 文件（在 `output/` 文件夹中）。
3.  **拖拽** JSON 文件到页面中央的虚线框内。

**操作方式**:
*   **左键拖拽**: 旋转视角
*   **右键拖拽**: 平移视角
*   **滚轮**: 缩放
*   **颜色含义**: 
    *   🟩 **绿色**: 低难度区域
    *   🟥 **红色**: 高难度区域
    *   🔴 **红色小球**: 未连接的开放出口（Dead Ends 或待扩展接口）

## ⚙️ 配置说明 (Configuration)

生成器依赖 `rail_config.csv` 来定义可用的轨道零件。该文件通常由 UE5 编辑器导出，包含以下关键信息：

*   **RowName**: 轨道的唯一标识符（通常包含尺寸信息，如 `_X1_Y1_Z1`）。
*   **Diff_Base / Difficulty**: 该轨道的基础难度系数。
*   **SizeX/Y/Z**: 轨道的逻辑占用尺寸（Grid Unit）。
*   **Exit_Array / Exits**: 轨道出口的位置和旋转信息。
*   **Type**: 轨道类型（Start, End, Normal）。

## 🧠 算法原理 (Algorithm)

本项目采用 **难度驱动的生长算法 (Difficulty-Driven Growth)**。

1.  **初始化**: 随机放置一个 Start 轨道，将其出口加入 `OpenList`。
2.  **生长循环**:
    *   从 `OpenList` 中取出一个可用接口。
    *   根据当前累计难度决定生成策略（正常生长 vs 寻找终点）。
    *   随机选择一个适配的轨道零件。
    *   进行 **碰撞检测** (AABB) 和 **边界检查**。
    *   如果放置成功，计算新难度并将其新出口加入 `OpenList`。
3.  **终结**: 当累计难度达到设定阈值 (`TARGET_DIFFICULTY`) 时，强制尝试放置 End 轨道。

## 🏗️ 代码结构与逻辑 (Code Structure & Logic)

核心脚本 `maze_generator.py` 采用严格的分层注释结构，逻辑流如下：

*   **1. 基础配置与常量**
    *   1.1 全局常量 (Scale, Bounds)
    *   1.2 边界模式枚举
*   **2. 核心数据结构**
    *   2.1 向量类 (`Vector3`)
    *   2.2 轨道配置项 (`RailConfigItem`)
    *   2.3 开放连接点 (`OpenConnector`)
    *   2.4 轨道实例 (`RailInstance`)
*   **3. 辅助计算逻辑**
    *   3.1 占用格子计算 (`calculate_occupied_cells`)
        *   3.1.1 确定局部遍历范围
        *   3.1.2 解析 Y 轴范围
        *   3.1.3 解析 Z 轴范围
        *   3.1.4 遍历并生成世界坐标
*   **4. 配置加载模块**
    *   4.1 加载配置 (`load_config`)
*   **5. 迷宫生成器核心 (`MazeGenerator`)**
    *   5.1 初始化
    *   5.2 状态检查与工具 (边界、碰撞、方向转换)
    *   5.3 核心放置逻辑 (`place_rail_v2`)
    *   5.4 生成流程 (`generate`)
    *   5.5 导出逻辑 (`export_json`)
*   **6. 程序入口**
    *   6.1 主执行块

可视化工具 `maze_viewer.html` 同样遵循分层结构：

*   **1. 基础配置与依赖**
    *   1.1 全局常量 (Grid Scale, Bounds)
*   **2. 场景初始化**
    *   2.1 调试辅助网格 (Debug Grid)
    *   2.2 坐标轴辅助
    *   2.3 相机与渲染器
    *   2.4 灯光系统
*   **3. 拖拽与文件解析**
    *   3.1 拖拽事件绑定
    *   3.2 JSON 解析
*   **4. 资源管理**
    *   4.1 共享几何体与材质 (InstancedMesh 优化)
    *   4.2 资源清理函数
    *   4.3 全局状态映射
*   **5. 场景构建核心 (`buildScene`)**
    *   5.1 清理旧场景
    *   5.2 更新 UI 统计信息
    *   5.3 绘制迷宫边界框
    *   5.4 实例化网格准备
    *   5.5 创建实例化网格
    *   5.6 填充实例数据 (Blocks, Arrows, Text)
*   **6. 交互与射线检测**
    *   6.1 射线初始化
    *   6.2 鼠标移动处理 (Hover 高亮)
*   **7. 渲染循环**

## 📄 输出格式 (Output Format)

生成的 JSON 文件包含以下核心字段：

*   **MapMeta**: 地图元数据（种子、难度、边界等）。
*   **Rail**: 放置的轨道列表。
    *   `Index`: 全局唯一索引。
    *   `Name`: 对应 CSV 中的 RowName。
    *   `Pos_Abs`: 物理世界坐标 (cm)。
    *   `Pos_Rev`: 逻辑网格坐标 (grid)。
    *   `Rot_Index`: 旋转索引 (0-3, 对应 0°-270°)。
    *   `Prev_Index` / `Next_Index`: 链表结构的连接关系。

## 🤝 贡献 (Contributing)

欢迎提交 Issue 或 Pull Request 来改进算法效率或增加新功能。

## 📜 许可证 (License)

[MIT License](LICENSE)
