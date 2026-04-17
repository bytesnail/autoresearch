# analyze_experiments.py 技术解读

> 2438 行 · 输入：`results.tsv` + git 历史 · 输出：`experiment_analysis.html` + `experiment_data.json`

---

## 1 数据采集

### 1.1 数据源

脚本从两个数据源采集信息：

**results.tsv** — 334 轮实验的结构化记录，每行 5 列，以 Tab 分隔：

```
<commit_hash>  <val_bpb>  <memory_gb>  <status>  <description>
e23b26d        1.260177   2.4          keep      baseline experiment
```

- `commit_hash`：7 位短哈希，指向该实验对应的代码快照
- `val_bpb`：验证集 Bits Per Byte，模型质量的核心指标（越低越好）；`CRASH` 或 `0.000000` 表示训练崩溃
- `memory_gb`：峰值显存占用（GB）
- `status`：实验决策 — `keep` / `NEW BEST` / `improved`（保留）、`discard` / `worse`（丢弃）、`crash`（崩溃）
- `description`：人类编写的实验描述，包含修改内容、比较对象、结论

**git 仓库** — 代码变更历史：

- `git log e23b26d..HEAD --format="%H|%s|%ai"` 获取从起始 commit 到 HEAD 的所有提交（hash / message / date）
- `git show -s --format=%ai <ref>` 分别获取起始和结束 commit 的作者日期，计算实验总运行时长

### 1.2 TSV 解析流程

解析函数 `parse_results(filepath)` 执行以下步骤：

**① 读取与拆行**

使用 `csv.reader(delimiter="\t")` 逐行读取。每行预期 5 个字段，但存在数据异常：某些行的两个实验被拼接在一起（缺少换行符），产生 9 个字段。解析器对此做了防御处理：

- 字段数 = 5 → 正常单行
- 字段数 ≥ 9 → 判定为两个实验合并，拆分为 `[row[:5], row[5:]]` 两段分别解析
- 字段数在 5-9 之间 → 取前 5 个字段，尝试从 description 末尾剥离附着的 commit hash

**② 字段解析**

```
val_bpb:   "CRASH" / "0.000000" → 标记为崩溃，val_bpb 设为 None 或 0.0
           其他无法转为 float 的值 → 同样标记为崩溃
           正常数值 → 直接使用

memory_gb: float 转换失败 → None

status:    原样保留 TSV 中的值（keep / discard / crash / NEW BEST 等）
```

**③ 基线与新纪录追踪**

- 第一个实验的 `val_bpb` 作为基线值（1.260177）
- 维护 `current_best` 变量：遍历过程中，若某实验 `val_bpb < current_best` 且非崩溃，则更新 `current_best` 并将该实验标记为 `is_best=True`
- 构建 `best_history` 列表（长度 = 实验总数），第 i 个元素记录截至第 i 个实验时的全局最佳 val_bpb

**④ 附加标记**

从 description 字段中提取额外信息：

- `lost_steps`：描述中包含 "lost steps" / "lost " / " steps)" 时为 True
- `steps_count`：正则匹配 `(N steps)` 或 `N steps vs` 提取训练步数
- `is_oom`：显存 > 10GB 且状态为 discard 时为 True（推断为 OOM）

**⑤ 输出**

```
experiments:       List[Experiment]  — 334 个实验记录
baseline_val_bpb:  float             — 1.260177
best_history:      List[float]       — 长度 334，累积最佳曲线
```

### 1.3 Experiment 数据结构

```python
@dataclass
class Experiment:
    index: int                # 顺序编号（0-333）
    commit: str               # commit 短哈希
    val_bpb: Optional[float]  # 验证 BPB（崩溃为 None 或 0.0）
    memory_gb: Optional[float]
    status: str               # keep / discard / crash / NEW BEST / improved / worse
    description: str          # 实验描述原文
    phase: str = ""           # 后续填充：研究阶段
    category: str = ""        # 后续填充：超参数类别
    is_best: bool = False     # 是否刷新全局最佳
    is_timeout: bool = False
    is_oom: bool = False
    lost_steps: bool = False
    steps_count: Optional[int] = None
```

---

## 2 数据分析

### 2.1 超参数分类

函数 `classify_hyperparameter(desc)` 将实验描述映射到 19 个类别。采用 if-elif 链实现优先级匹配：

```
Architecture → Learning Rate (Matrix) → Learning Rate (Scalar)
→ Learning Rate (Embedding) → Learning Rate (Unembedding)
→ Learning Rate (Other) → Weight Decay → Optimizer (Muon)
→ Optimizer (Adam) → Softcap → LR Schedule → Final LR Fraction
→ Batch Size → Weight Tying → Initialization → Normalization
→ Value Embedding → Regularization → Compilation/Performance
→ RoPE → WD Masking → Baseline → Other
```

匹配规则基于 description 中的关键词子串（如 `"matrix lr" in desc_lower`）。优先级设计确保语义更具体的类别优先匹配——例如含 "matrix lr" 的描述归入 "Learning Rate (Matrix)" 而非更宽泛的 "Learning Rate (Other)"。

### 2.2 研究阶段分类

函数 `classify_phase(experiments)` 按实验序号的硬编码区间将 334 个实验分为 10 个阶段：

| 阶段 | 序号范围 | 主题 |
|------|---------|------|
| Phase 1 | 0–8 | 初始探索 |
| Phase 2 | 9–42 | 基础超参搜索 |
| Phase 3 | 43–68 | 动量与 WD 优化 |
| Phase 4 | 69–121 | 精细网格搜索 |
| Phase 5 | 122–131 | 权重绑定 |
| Phase 6 | 132–153 | QK Norm 与 Softcap 深挖 |
| Phase 7 | 154–207 | 系统性重测 |
| Phase 8 | 208–255 | 高级技术探索 |
| Phase 9 | 256–272 | WD 调度发现 |
| Phase 10 | 273+ | 最终优化 |

阶段划分是人工定义的，基于对实验内容时间顺序和主题的分析。

### 2.3 统计计算

函数 `compute_statistics(experiments, baseline_val_bpb)` 遍历所有实验，累计计算：

**全局统计**（`stats` 字典顶层）：

| 键 | 含义 |
|----|------|
| `total` | 总实验数（334） |
| `improved` | 改善次数（new_best + kept） |
| `worse` | 未改善次数（discard） |
| `crashes` | 崩溃次数 |
| `new_best` | 刷新全局最佳的次数（40，含基线） |
| `best_val_bpb` | 全局最佳 val_bpb（1.218771） |
| `improvement_from_baseline` | 相对基线的改善百分比（3.29%） |

**分类维度**（嵌套字典）：

- `category_stats[category_name]` → `{total, improved, worse, crash, best}`
- `phase_stats[phase_name]` → `{total, improved, worse, crash}`

一个实验的归类判定逻辑：

```
is_best=True 或 status="NEW BEST" 或 description 含 "new best"
    → new_best += 1, improved += 1
status="keep" 或 "improved"
    → improved += 1, kept += 1
status="discard" 或 "worse"
    → worse += 1, discarded += 1
status="crash"
    → crashes += 1
```

### 2.4 里程碑提取

函数 `extract_milestones(experiments, baseline_val_bpb)` 独立于 `parse_results` 中的 `is_best` 标记，重新遍历实验列表，找出每次刷新全局最佳的时刻。输出 39 个里程碑（不含基线本身），每个包含：

| 字段 | 含义 |
|------|------|
| `index` | 实验序号 |
| `commit` | commit hash |
| `val_bpb` | 该实验的 val_bpb |
| `description` | 实验描述 |
| `step_improvement` | 相对上一次最佳的改善百分比 |
| `total_improvement` | 相对基线的累计改善百分比 |

### 2.5 敏感性数据提取

4 类超参数敏感性分析使用统一的数据提取架构。每类有一个提取函数，返回 `[{"value": float, "val_bpb": float, "is_best": bool}]` 列表，按 `value` 排序。图表函数和 JSON 导出函数共享同一组提取函数作为数据源。

**Learning Rate**（`_extract_lr_sensitivity_data`）

将 LR 实验分为 4 类：`matrix_lr`、`scalar_lr`、`embedding_lr`、`unembedding_lr`。对每个实验的 description 执行 4 组正则匹配：

```python
r"(?:^|(?<=\s))matrix[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)"    # matrix_lr
r"(?:^|(?<=\s))scalar[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)"    # scalar_lr
r"unembedding[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)"             # unembedding_lr
r"(?:^|(?<=\s))embedding[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)"  # embedding_lr
```

特殊规则：当 description 同时含 "unembedding" 和 "embedding" 关键词时，unembedding 优先匹配，且命中后跳过 embedding 检查（`continue`）。

**Weight Decay / Softcap**（`_extract_param_sensitivity`，通用函数）

接受 `keyword_patterns`（关键词列表）和 `value_pattern`（正则）两个参数：

| 超参数 | keyword_patterns | value_pattern |
|--------|-----------------|---------------|
| Weight Decay | `["weight decay", "wd=", "wd_", "wd "]` | `r"(?:weight.?decay\|wd)[\s_=]+(\d+\.?\d*)"` |
| Softcap | `["softcap"]` | `r"softcap[\s_=]+(?:0\.\|)(\d+\.?\d*)"` |

Weight Decay 有额外处理：提取值 > 1.0 时除以 100（将百分比值归一化为小数）。

**Momentum**（`_extract_momentum_data`）

匹配模式：`r"momentum[\s_]+(?:end\s+|start\s+|constant\s+|ramp\s+)?(?:0\.)(\d+)"`。提取后通过 `float("0." + group)` 重构值（如匹配到 "85" → 0.85）。

---

## 3 数据保存

### 3.1 输出文件

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `experiment_analysis.html` | 自包含 HTML | ~1.6MB | 图表以 base64 PNG 内嵌，无外部依赖 |
| `experiment_data.json` | JSON | ~160KB | 结构化数据，供后续程序读取 |

### 3.2 JSON 结构

函数 `export_chart_data()` 聚合所有分析结果为一个字典：

```
{
  "meta": {
    "total_experiments": 334,
    "baseline_val_bpb": 1.260177,
    "best_val_bpb": 1.218771,
    "improvement_pct": 3.29,
    "improved": 40,
    "worse": 289,
    "crashes": 5,
    "start_date": "2026-04-16 02:05:12",
    "end_date": "2026-04-17 20:58:08",
    "duration": "1 天 18 小时 52 分钟"
  },
  "progress": [
    {"index": 0, "running_best": 1.260177, "improvement_pct": 0.0},
    ...  // 334 个元素
  ],
  "categories": {
    "Architecture": {"total": N, "improved": N, "worse": N, "crash": N, "new_best": N, "success_rate": N},
    ...  // 19 个类别
  },
  "phases": {
    "Phase 1: Initial Exploration": {"total": N, "improved": N, "worse": N, "crash": N, "success_rate": N, "best_val_bpb": N},
    ...  // 10 个阶段
  },
  "lr_sensitivity": {
    "matrix_lr": [{"value": N, "val_bpb": N, "is_best": bool}, ...],
    "scalar_lr": [...],
    "embedding_lr": [...],
    "unembedding_lr": [...]
  },
  "weight_decay": [...],
  "softcap": [...],
  "momentum": [...],
  "milestones": [
    {"index": N, "commit": "...", "val_bpb": N, "description": "...", "step_improvement_pct": N, "total_improvement_pct": N},
    ...  // 39 个里程碑
  ],
  "experiments": [
    {"index": N, "commit": "...", "val_bpb": N, "memory_gb": N, "status": "...", "description": "...", "phase": "...", "category": "...", "is_best": bool},
    ...  // 334 个实验
  ]
}
```

**去重规则**：若不同图表使用相同数据源，JSON 中只存储一份：

- Progress 折线图 + Cumulative improvement 面积图 → 共享 `progress` 键
- Category 饼图 + Success rate 柱状图 → 共享 `categories` 键

### 3.3 HTML 报告保存

报告通过 Python f-string 拼接生成（约 700 行 HTML 模板内嵌在 `generate_html_report` 函数中）。所有图表通过 `chart_to_base64(fig)` 函数转为 base64 编码的 PNG 图片，以 `data:image/png;base64,...` 格式直接内嵌到 `<img>` 标签。整个报告为单个 HTML 文件，无外部依赖，可直接在浏览器打开。

---

## 4 报告设计与编排

### 4.1 页面框架

```
┌─────────────────────────────────────────────────────┐
│  Navigation Bar (sticky, 深色背景)                    │
│  概览 | 进展 | 里程碑 | 分类 | 阶段 | 敏感性          │
│  崩溃 | 未改善 | 理论 | 演进 | Git | 全部实验         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  标题 + 副标题 + 时间横幅                             │
│                                                     │
│  §1  总体概览                                        │
│  §2  实验进展趋势                                     │
│  §3  关键里程碑                                       │
│  §4  超参数分类分析                                    │
│  §5  研究阶段演进                                     │
│  §6  超参数敏感性分析                                  │
│  §7  关键发现总结                                     │
│  §8  崩溃与超时分析                                   │
│  §9  未改善实验深度分析                                │
│  §10 理论解释与参考文献                                │
│  §11 修改内容演进                                     │
│  §12 Git 提交记录                                    │
│  §13 全部实验明细（可搜索）                             │
│                                                     │
│  Footer                                             │
└─────────────────────────────────────────────────────┘
```

导航栏使用 `position: sticky` 固定在页面顶部，点击各链接平滑滚动到对应章节。

### 4.2 时间横幅

紫色渐变横幅（`.time-banner`），三栏 Flexbox 布局：

| 左栏 | 中栏（加粗加大） | 右栏 |
|------|-----------------|------|
| 实验开始时间 | 总运行时长 | 实验结束时间 |

数据来源：`get_commit_date(repo_path, "e23b26d")` 和 `get_commit_date(repo_path, "HEAD")` 获取作者日期，`format_duration()` 计算时长（天/小时/分钟）。

### 4.3 各章节内容

#### §1 总体概览

6 张统计卡片（CSS Grid 自适应布局）：

| 卡片 | 数值来源 | 样式 |
|------|---------|------|
| 总实验数 | `stats["total"]` | 默认深色 |
| 改善次数 | `stats["new_best"]` | 绿色 |
| 未改善/丢弃 | `stats["worse"]` | 红色数值 |
| 运行崩溃 | `stats["crashes"]` | 橙色 |
| 总改善幅度 | `stats["improvement_from_baseline"]` | 蓝色 |
| 最佳 Val BPB | `stats["best_val_bpb"]` | 绿色 |

其后是一张状态分布饼图（8×8 英寸），展示 Improved / Discarded / Crash 三类比例。

#### §2 实验进展趋势

**Val BPB 进展图**（`generate_progress_chart`，18×7 英寸双面板）：

- 左面板（主图，宽高比 2.5:1）：散点图 + 运行最佳线
  - 绿色散点 = 新纪录实验
  - 蓝色散点 = 保留/改善实验
  - 红色散点 = 丢弃/恶化实验
  - 绿色实线 = `best_history` 累积最佳曲线
  - 橙色虚线 = 基线 val_bpb
  - 图例区分 5 类元素
- 右面板（缩放图）：聚焦 val_bpb 最佳区域（y 轴范围：全局最小值 - 0.002 到基线值），标注最终最佳值

**累积改善图**（`generate_cumulative_improvement`，14×6 英寸）：

- 绿色填充面积图：x 轴为实验序号，y 轴为相对基线的改善百分比
- 标注前 15 个里程碑（箭头 + 描述文字）
- y 轴格式化为百分比

#### §3 关键里程碑

HTML 表格列出 39 个里程碑，列包括：序号、Commit（8 位短哈希、`<code>` 标签）、Val BPB（6 位小数）、描述、步骤改善（绿色加粗、带 `-` 前缀）、累计改善。

#### §4 超参数分类分析

三部分内容：

1. **环形饼图**（`generate_category_pie`，13×8 英寸）：各类别实验数量分布。内环半径 65%，外环白色描边，右侧图例显示类别名和数量。
2. **堆叠柱状图**（`generate_success_rate_by_category`，15×7 英寸）：各类别的 Improved（绿）/ Worse（红）/ Crash（橙）堆叠柱，顶部标注成功率百分比。仅显示实验数 ≥ 3 的类别。
3. **分类统计表**：每个类别的总数、改善数、未改善数、崩溃数、新纪录数、成功率。

#### §5 研究阶段演进

1. **水平柱状图**（`generate_phase_progress`，14×7 英寸）：10 个阶段的最佳 val_bpb。颜色通过 RdYlGn_r 色图映射（红=差，绿=好），柱顶标注精确数值。标注全局最佳值。
2. **阶段统计表**：每个阶段的实验数、改善数、未改善数、崩溃数、成功率。
3. **10 张阶段说明卡片**：白色背景卡片（`.evolution-item`），描述每个阶段的主题、关键发现和超参数变化路径。

#### §6 超参数敏感性分析

4 张图表纵向排列：

**LR 敏感性**（`generate_lr_sensitivity`，16×11 英寸 2×2 子图）

每个子图展示一类 LR（Matrix / Scalar / Embedding / Unembedding）的值与 val_bpb 的关系：
- 散点大小区分：新纪录实验 80px / 非新纪录 30px
- 颜色区分：新纪录绿色 / 非新纪录蓝色
- 数据量 ≥ 4 时叠加二次多项式拟合曲线（红色虚线）
- 标注最佳点（val_bpb 最低）的坐标

**Weight Decay / Softcap / Momentum 散点图**

三张图表采用相同的可视化模式（12-13×7 英寸）：
- 按参数值分组，每组内散点带固定种子随机抖动（避免重叠）
- 绿色 = 组内有新纪录实验 / 红色 = 无新纪录
- 每组叠加均值水平线（深色）
- WD 图额外叠加全局最佳值水平参考线（绿色点线）

#### §7 关键发现总结

表格列出 8 个最具影响力的发现，按影响力排序：移除 QK 归一化（大）、权重绑定（大）、Softcap 优化（中）、延迟 WD（中）等。

#### §8 崩溃与超时分析

- 崩溃实验表格：序号、commit、描述（黄色背景行）
- 崩溃原因分类卡片：形状不匹配、梯度缺失、CUDA 编译错误
- 超时与步数损失分析卡片：OOM、步数减少、步数损失、编译超时 4 类

#### §9 未改善实验深度分析

统计未改善实验总数，按原因分类的表格：超参近最优边界（~120）、重测确认（~40）、架构不适合小模型（~15）等。

#### §10 理论解释

9 个理论卡片（`.theory-card`，白色背景圆角卡片），每个包含：
- 实验现象描述
- 理论解释（多个小节，加粗标题）
- 参考文献列表（灰色斜体）

覆盖主题：ReLU² vs GELU/SiLU、Muon 优化器、Attention Logit Softcap、权重绑定、移除 QK 归一化、延迟权重衰减、PE Norm Constant、Full Attention、关键失败尝试。

#### §11 修改内容演进

6 个演进卡片，分别描述学习率优化（~35%）、正则化策略（~20%）、优化器配置（~15%）、架构改动（~10%）、学习率调度（~10%）、其他探索（~10%）的参数变化路径。

#### §12 Git 提交记录

表格展示从起始 commit 到 HEAD 的所有提交：8 位短哈希（`<code>` 标签）、时间、描述。

#### §13 全部实验明细

带搜索框的完整实验表格。搜索框绑定 `input` 事件，实时过滤表格行（JavaScript `textContent.toLowerCase().includes(filter)`）。

表格列：序号、Commit、Val BPB、内存(GB)、状态、类别、描述。行背景色按状态区分：新纪录（浅绿）、保留（淡绿）、丢弃（浅红）、崩溃（浅黄）。

### 4.4 CSS 设计要点

- **布局**：CSS Grid（统计卡片、响应式 2 列）+ Flexbox（时间横幅、导航栏、阶段时间线）
- **配色**：主色调 `#2c3e50`（深蓝灰），改善绿 `#27ae60`/`#2ecc71`，恶化红 `#e74c3c`，崩溃橙 `#e67e22`
- **表格**：圆角阴影卡片样式，深色表头，行悬停高亮
- **响应式**：`@media (max-width: 768px)` 切换为 2 列网格，缩小字号和间距
- **导航**：`position: sticky; top: 0; z-index: 100` 固定顶部

### 4.5 图表生成管线

所有图表遵循统一流程：

```
matplotlib Figure → fig.savefig(BytesIO, format="png", dpi=150) → base64.b64encode → HTML <img src="data:...">
```

样式统一通过 `_apply_style(ax, title, xlabel, ylabel)` 函数处理：隐藏上/右边框、浅灰色网格（alpha=0.15）、标题 13px 加粗深蓝、轴标签 11px 灰色。

---

## 5 主流程

`main()` 函数按以下顺序执行：

```
1. parse_results("results.tsv")
   → experiments[334], baseline_val_bpb, best_history[334]

2. classify_phase(experiments)           → 为每个 exp 赋值 phase
   classify_hyperparameter(desc) × 334   → 为每个 exp 赋值 category

3. compute_statistics(experiments, baseline)
   → stats dict

4. extract_milestones(experiments, baseline)
   → milestones[39]

5. 生成 10 张图表（各 generate_* 函数）
   → charts dict (10 个 base64 PNG 字符串)

6. parse_git_history(repo_path, "e23b26d")
   get_commit_date × 2 + format_duration
   → git_commits, start_date, end_date, duration_str

7. generate_html_report(...)
   → 写入 experiment_analysis.html

8. export_chart_data(...)
   → 写入 experiment_data.json
```

步骤 5 中的敏感性图表调用 `_extract_*` 提取函数获取数据，步骤 8 的 JSON 导出同样调用同一组提取函数。两者共享数据提取逻辑，无重复实现。
