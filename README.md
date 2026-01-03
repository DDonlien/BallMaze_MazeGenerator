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
├── pseudo.md                  # [文档] 算法逻辑流程图 (Mermaid)
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

> 详细的逻辑流程图请查阅 [pseudo.md](pseudo.md)。

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
