# Action 数据可视化计划

## TL;DR

> **快速摘要**: 在 `scripts/dual_robot/data_vis.py` 中添加代码，将 actions 数据的 7 个维度分别画成曲线图并保存为独立文件。

> **输出文件**: `action_dim_0.png` ~ `action_dim_6.png`

> **Estimated Effort**: Short (单一脚本修改)

---

## Context

### 原始请求
用户想要在 `scripts/dual_robot/data_vis.py` 中可视化动作数据，把 actions 的每一维度都单独画成曲线图，保存成单独的文件。

### 需求确认
- **文件命名**: `action_dim_X.png` (X = 0-6)
- **数据维度**: 7 维
- **图表风格**: 简洁风格，不需要标题和坐标轴标签
- **仅可视化 actions**: 不包含 states 数据

### 技术信息
- 数据来源: `/home/amax/workspace/dataset/lerobot/pick_up_the_beaker/data/chunk-000/`
- 数据获取: `np.stack(data["action"].to_numpy())`
- 已有依赖: matplotlib.pyplot, numpy, pandas

---

## Work Objectives

### Core Objective
在 `data_vis.py` 中实现 action 各维度的曲线图绘制和保存功能。

### Concrete Deliverables
- 修改 `scripts/dual_robot/data_vis.py`，添加可视化代码
- 生成 7 个图像文件: `action_dim_0.png` ~ `action_dim_6.png`

### Definition of Done
- [ ] 运行 `python scripts/dual_robot/data_vis.py` 成功执行
- [ ] 在数据目录生成 7 个 `action_dim_X.png` 文件
- [ ] 每张图显示对应维度的动作曲线

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (本地调试脚本，无需测试框架)
- **Automated tests**: NO
- **Agent-Executed QA**: YES - 手动执行脚本验证输出文件

---

## Execution Strategy

### Task Breakdown

这个任务简单直接，无需分波执行。

---

## TODOs

- [ ] 1. 修改 data_vis.py，添加 action 可视化代码

  **What to do**:
  - 遍历 actions 的 7 个维度
  - 对每个维度绘制折线图
  - 保存为 `action_dim_{i}.png` 文件

  **Must NOT do**:
  - 不修改数据加载逻辑（保持现有代码不变）
  - 不添加不必要的依赖

  **Recommended Agent Profile**:
  - **Category**: `quick` - 简单代码修改任务
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: N/A (单一任务)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `scripts/dual_robot/data_vis.py:14-16` - 数据加载代码参考

  **Acceptance Criteria**:
  - [ ] 脚本执行无报错
  - [ ] 生成 7 个 PNG 文件

  **QA Scenarios**:

  Scenario: 运行脚本生成可视化图像
    Tool: Bash
    Preconditions: 数据文件存在于 `/home/amax/workspace/dataset/lerobot/pick_up_the_beaker/data/chunk-000/`
    Steps:
      1. 执行 `cd /home/amax/workspace/lerobot && python scripts/dual_robot/data_vis.py`
      2. 检查输出目录是否生成文件
    Expected Result: 脚本正常退出，生成 7 个 action_dim_X.png 文件
    Evidence: ls 命令查看生成的文件列表

---

## Success Criteria

### Verification Commands
```bash
python scripts/dual_robot/data_vis.py
ls -la /home/amax/workspace/dataset/lerobot/pick_up_the_beaker/data/chunk-000/action_dim_*.png
```

### Final Checklist
- [ ] 7 个 action_dim_X.png 文件生成
- [ ] 脚本无报错执行
