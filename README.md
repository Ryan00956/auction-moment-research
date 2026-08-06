# 竞拍时刻：公开研究档案与实验性 OCR 助手

这是一个面向《碧蓝航线》“竞拍时刻”玩法的公开研究档案，包含脱敏文本数据、训练与分析代码、冻结模型，以及一个实验性的 OCR/视觉助手原型。

> [!IMPORTANT]
> “竞拍时刻”活动已经下线，本项目自 **2026-08-06** 起进入暂时休眠状态。仓库保留用于数据研究、结果复现和未来可能返场时的重新验证，目前不再作为可用的实时活动工具维护。

> [!WARNING]
> 公开实时助手没有在正式活动环境中完成过一整局从开局、逐轮识别与推理到对局结束的端到端跑通验证。现有证据主要覆盖组件测试、冻结数据上的模型研究、脱敏数据发布和界面原型验证。它不是生产系统，预测未经新时间块前瞻校准，所有结果均为人工参考且 `actionable=false`。

项目状态和返场条件分别见 [`STATUS.md`](STATUS.md) 与 [`RETURN_RUNBOOK.md`](RETURN_RUNBOOK.md)，归档资产说明见 [`archive/season-2026-archive-v1.md`](archive/season-2026-archive-v1.md)。

> 冻结助手版本：[latest-model-v6-v2.0.0-beta.6](https://github.com/Ryan00956/auction-moment-research/releases/tag/latest-model-v6-v2.0.0-beta.6)；公开季末快照：[season-2026-archive-v1](https://github.com/Ryan00956/auction-moment-research/releases/tag/season-2026-archive-v1)。

## 已公开内容

这个仓库首先提供可核验的研究材料：

- `auction-moment-public-v1` 脱敏文本数据及逐文件哈希；
- 世界模型、对手行为聚类和统计分析代码；
- 冻结实验结果、模型卡、数据卡和分析报告；
- v6 价值模型、world-model-v2 与三个视觉模型的版本化发布资产；
- 用于继续研究和界面实验的 OCR/视觉助手原型。

## 实验性助手包含的组件

下面列出的是已经实现的原型组件，不代表它们已经在一整局正式活动中端到端跑通：

- **自动监视对局**：大厅待机，检测开局后进入轮次状态机，每轮自动扫描一次地图并生成预测。
- **OCR 事件识别**：识别轮次、公共事件、个人事件和持有金额，支持逐项人工修改。
- **YOLO 地图识别**：识别藏品位置、完整形状、品质、规格和完整身份候选。
- **可视化地图建模**：在网格上拖拽新增、移动或删除藏品框，快速补充 OCR 漏掉的信息。
- **中文藏品图鉴**：内置 120 个藏品名称和预览图，可按品质与尺寸筛选并确认身份。
- **v6 + world-model-v2 联合推理**：结合当前事件、地图证据和概率世界，输出 P10、P50、P90 估值区间。
- **建议最高出价**：默认按保守估值 `P10 × 90%` 给出人工参考值，并显示对应对手最高报价。
- **条件冲突诊断**：条件不兼容时指出造成冲突的事件或地图证据；有限样本耗尽时会自动进行条件重采样。
- **完整运行日志**：日志区支持滚动、选择复制和一键复制全部，可查看 OCR、地图快照、推理状态和异常详情。

## 原型界面

界面分为左右两部分：

- **左侧：地图建模与快速纠错**
  - 在网格上查看自动识别的藏品位置。
  - 切换“左上角位置”或“完整形状”。
  - 设置无品质、白、蓝、紫、金、彩等品质。
  - 设置 1×1 至 3×3 的规格。
  - 从中文图鉴和预览图中确认完整身份。
- **右侧：事件、证据、预测与日志**
  - 修改公共事件和个人事件。
  - 修正持有金额、地图行数和轮次。
  - 查看 v6 + world-model-v2 的估值区间与建议出价。
  - 复制完整诊断日志，便于反馈和排查。

状态机被设计为执行下面的流程，但该完整链路尚未在正式活动环境完成一整局验证：

```text
大厅等待 → 检测开局 → 识别事件 → 扫描本轮地图 → 生成预测 → 等待下一轮
```

## 复现实验性助手

活动已下线，以下步骤只用于复现、审阅或未来返场后的重新验证，不构成当前可用性承诺。优先使用 `season-2026-archive-v1` 的固定资产和依赖清单，不要将 `main` 分支的未来变化与冻结模型混用。

### 1. 准备环境

推荐环境：

- Windows 10 或 Windows 11
- Python 3.11 或 Python 3.12
- Git
- 可用的 ADB
- 模拟器使用 `1280×720` 横屏分辨率

程序会优先使用 CUDA；没有可用显卡时也可以使用 CPU，只是视觉识别速度会慢一些。

### 2. 下载并安装

```powershell
git clone https://github.com/Ryan00956/auction-moment-research.git
cd auction-moment-research

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e .\apps\ephemeral-assistant
```

首次安装需要下载 OCR、YOLO、OpenCV 和 PyTorch 等依赖，耗时取决于网络环境。

### 3. 检查模拟器连接

```powershell
adb devices
```

确认目标设备显示为 `device`。如果同时连接了多个设备，记下目标设备编号，启动时通过 `--serial` 指定。

### 4. 启动助手

第一次运行可自动下载并校验五个模型文件：

```powershell
.\.venv\Scripts\auction-vision-assistant.exe `
  --download-models `
  --models .\models\latest-model-v6-v2.0.0-beta.6 `
  --treasures .\data\v1\core\treasures.csv
```

多设备环境示例：

```powershell
.\.venv\Scripts\auction-vision-assistant.exe `
  --download-models `
  --models .\models\latest-model-v6-v2.0.0-beta.6 `
  --treasures .\data\v1\core\treasures.csv `
  --serial 127.0.0.1:7555
```

强制使用 CPU：

```powershell
.\.venv\Scripts\auction-vision-assistant.exe `
  --download-models `
  --models .\models\latest-model-v6-v2.0.0-beta.6 `
  --treasures .\data\v1\core\treasures.csv `
  --device cpu
```

模型下载完成后，后续启动可以省略 `--download-models`。

## 人工校正方法

### 修正事件

1. 在右侧事件列表选中公共事件或个人事件。
2. 在下方输入正确的事件文本。
3. 点击“应用修正”。
4. 系统会基于新事件重新计算预测。

### 修正地图

1. 在左侧选择“左上角位置”或“完整形状”。
2. 选择品质；完整形状还可以选择规格。
3. 在地图空白处拖拽新增藏品框。
4. 拖动已有框可以移动，右键可以删除。
5. 选中已有框后，可修改品质、规格并点击“应用到选中框”。

支持的地图证据包括：

- 已知左上位置，品质未知或品质已知；
- 已知完整位置与形状，品质未知或品质已知；
- 已知完整身份。

### 确认完整身份

1. 先在地图上选中一个藏品框。
2. 在右侧图鉴中搜索或筛选藏品。
3. 查看名称、品质、规格、价值和预览图。
4. 点击“确认所选身份并套用品质/尺寸”。

自动 OCR 达到置信度且与图鉴品质、尺寸一致时会直接采用；人工指定完整身份时需要明确确认。

## 如何理解预测结果

| 字段 | 含义 |
|---|---|
| P10 | 偏保守的总价值估计，约有 90% 的模拟结果高于它 |
| P50 | 中位数估值 |
| P90 | 偏乐观的总价值估计 |
| 建议最高出价 | 默认使用 `P10 × 90%` 计算的人工参考上限 |
| 对应对手最高报价 | 根据本轮规则反推的对手报价上限 |
| 兼容世界 | 同时满足当前事件和地图证据的模拟世界数量 |
| 状态/诊断 | 当前是否缺少证据、发生条件冲突或使用了回退结果 |

建议先查看状态和诊断，再看具体数值。OCR 未完成、事件无法解析或地图条件冲突时，界面会说明缺少什么，而不是只显示一个看似精确的数字。

本项目只负责识别、估值和提供人工参考，不会替用户执行出价。

## 开源内容

### 实时助手

[`apps/ephemeral-assistant`](apps/ephemeral-assistant) 包含：

- 对局视觉状态机；
- RapidOCR 固定区域识别；
- 三个 YOLO 模型的运行适配；
- 地图人工校正器；
- v6 + world-model-v2 推理适配；
- 预测、建议出价和诊断界面。

### 最新模型

GitHub Release 提供可直接使用的五个模型文件：

| 模型 | 用途 |
|---|---|
| `latest-value-model-v6.joblib` | 最新冻结 v6 价值模型 |
| `world-model-v2-S.json` | S 档概率世界模型 |
| `clue-spatial-yolo26n-v1.pt` | 地图位置与形状识别 |
| `clue-detector-yolo26n-v1.pt` | 品质与属性识别 |
| `treasure-classifier-yolo26n-v1.pt` | 完整藏品身份分类 |

模型说明见 [`models/LATEST_MODEL_CARD.md`](models/LATEST_MODEL_CARD.md)。

### 公开研究数据

当前公开数据版本为 `auction-moment-public-v1`：

| 表 | 行数 | 内容 |
|---|---:|---|
| `sessions.csv` | 1,816 | 场次级汇总 |
| `rounds.csv` | 9,080 | 五轮结构化明细 |
| `opponent_bids.csv` | 21,632 | 对手报价记录 |
| `treasures.csv` | 58,068 | 藏品与地图特征 |
| `event_catalog.csv` | 46 | 已观察事件目录 |
| `event_instances.csv` | 9,326 | 事件实例 |
| `event_offers.csv` | 18,664 | 个人事件候选 |
| `player_rounds.jsonl` | 13,656 | 玩家—场次—轮次研究样本 |

数据字段和版本说明见 [`data/v1/DATA_CARD.md`](data/v1/DATA_CARD.md)。

### 分析结论

- [藏品生成规律](reports/generation-patterns-v1.md)
- [公共事件与个人事件刷新规律](reports/event-refresh-patterns-v1.md)
- [对手行为原型研究](reports/opponent-archetypes-v1.md)
- [对手行为无监督聚类](reports/opponent-clustering-v1.md)

当前研究表明，完整五轮轨迹可以形成两个较稳定的描述簇；第二轮结束后的前缀特征对第五轮高报价判断有一定改善，但相对简单状态基线的额外收益仍未达到稳定显著。因此，对手聚类目前作为研究结果开放，还没有直接加入实时出价特征。

## 复现和继续研究

项目的主要目录：

```text
apps/ephemeral-assistant/   实时 OCR、YOLO、地图校正与预测界面
data/v1/                   公开研究数据
models/                    模型卡、发布清单与生成说明
reports/                   中文分析报告
results/                   冻结实验结果
scripts/                   发布、验证与复现工具
src/auction_moment_research/ 世界模型、聚类与数据处理代码
tests/                     研究代码测试
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest apps\ephemeral-assistant\tests -q
.\.venv\Scripts\python.exe scripts\verify_public_release.py
```

重新训练公开世界模型：

```powershell
.\.venv\Scripts\python.exe -m auction_moment_research.train_world_model `
  --input .\data\v1\core\treasures.csv `
  --output .\models\generated\world-model-v1.json `
  --manifest .\models\generated\manifest.json
```

重新生成对手聚类研究：

```powershell
.\.venv\Scripts\python.exe -m auction_moment_research.opponent_clustering `
  --research-root .\data\v1\opponents `
  --output-root .\results\reproduced-opponent-clustering `
  --overwrite
```

仓库休眠期间不承诺处理 Issue、功能请求或 Pull Request。未来若活动返场，应先按 [`RETURN_RUNBOOK.md`](RETURN_RUNBOOK.md) 完成规则、视觉、数据和前瞻验证，再恢复开发。

## 常见问题

### 启动后看不到左侧地图

请确认使用的是 beta.6 或更新版本。beta.5 存在首次布局可能把地图压缩到 1 像素的问题，beta.6 已修复。

### 提示找不到 ADB 设备

先运行 `adb devices`。如果设备未出现，请检查模拟器的 ADB 开关；如果有多个设备，请通过 `--serial` 指定。

### 一直没有给出预测

查看右下角诊断日志和预测状态。常见原因包括事件尚未识别、持有金额缺失、地图扫描尚未完成或当前证据互相冲突。

### OCR 识别错了怎么办

事件可以在右侧列表直接修改；地图可以在左侧重新框选；完整身份可以从中文图鉴中搜索并确认。修改后会自动触发新一轮推理。

### 没有 NVIDIA 显卡能运行吗

可以。默认 `--device auto` 会自动选择 CUDA 或 CPU，也可以显式传入 `--device cpu`。CPU 模式的地图扫描速度会更慢。

## 项目状态

- 状态：`HIBERNATING`，自 2026-08-06 起暂时休眠；
- 冻结助手版本：`0.2.0b6`；
- 完整正式对局端到端验证：**未完成**；
- 新时间块前瞻盲测和生产启用：**未完成**；
- 自动出价：**不存在于公开项目中**；
- 主要用途：模型研究、公开数据、分析复现和未来返场时的验证起点。

估值结果来自已冻结样本和概率世界模拟，不应被理解为确定收益承诺。更完整的已验证/未验证能力矩阵见 [`STATUS.md`](STATUS.md)。

## 许可证

- 研究代码：Apache License 2.0，见 [`LICENSE`](LICENSE)。
- 实时助手：AGPL-3.0-only，见 [`apps/ephemeral-assistant/LICENSE`](apps/ephemeral-assistant/LICENSE)。
- YOLO 模型资产：AGPL-3.0，详见 [`models/VISION_MODEL_CARD.md`](models/VISION_MODEL_CARD.md)。
- 数据、报告与结果：CC BY 4.0，见 [`DATA_LICENSE.md`](DATA_LICENSE.md)。
- 第三方名称、商标和图鉴图片的权利说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
