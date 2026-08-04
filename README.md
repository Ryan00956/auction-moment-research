# Auction Moment Research

这是一个从私有采集工程中独立导出的公开研究仓库，包含：

- 匿名化、纯文本、带哈希清单的数据集；
- 可从公开数据重新生成的概率世界模型和对手行为聚类代码；
- 冻结样本上的分析结论、评估指标和不确定性边界；
- 数据匿名化、模型训练和公开发布检查工具；
- 不抓包、不保存会话、可调用最新 v6 + world-model-v2 的 OCR/YOLO 实时研究助手。

仓库不包含图片、视频、抓包、协议解析、解密脚本、游戏二进制、玩家身份、
自动点击或自动出价执行层。实时助手只允许两种固定的左侧地图滚动手势，不具备
任意点击、输入或报价接口。Git 历史保持文本-only；经过路径净化和参数等价验证的
中间态视觉权重和经过脱敏等价检查的最新估值模型通过 GitHub Release 单独发布。助手只显示未校准研究估计，不会
替用户提交报价。

## 无持久化视觉助手（beta）

[`apps/ephemeral-assistant`](apps/ephemeral-assistant) 是一个独立 AGPL-3.0
子项目：

- 状态机常驻监视 `1280×720` 画面：大厅等待，检测开局，每轮自动扫描一次，
  输出预测后待机，终局清空；
- ADB 除截图外，只能执行代码中固定的地图归顶/下滚手势；不发送点击、按键、
  报价、抓包或任意坐标输入；
- RapidOCR 识别轮次、公共/个人事件和持有金额；
- 三份 YOLO 权重识别位置、品质、规格和完整揭示身份 Top-k；达到阈值的自动
  完整身份直接采用，仍可在画布上人工改错；
- OCR 转义层把已解析事件和显式标注来源的地图/时点证据送入最新冻结
  `v6 + world-model-v2`；自动滚动行数始终标为估算而不是精确证明；
- 每轮显示 P10/P50/P90，并以 `P10 × 0.90` 给出仅供人工参考的建议最高出价；
  初始有限世界集合归零时先按全部条件额外生成最多 32,768 个世界，仍无结果才
  保留 v6 保守回退并指出导致粒子耗尽的事件或地图条件；
- 左侧地图画布恢复原校对器的 13 类证据：左上角位置/完整形状分别组合
  无品质、白、蓝、紫、金、彩，再加完整身份；标注框几何与真实尺寸分离，
  人工完整身份必须从兼容图鉴列表明确选中；右侧事件可逐轮修正；
- 人工修正按 revision 锁定，后续 OCR 不会静默覆盖，过期推理结果不会显示；
- 完整身份选择显示公开中文图鉴名称与 80x80 预览图，不再显示内部编号；
- 可滚动、可复制的内存诊断日志保留 OCR 语义、地图快照、推理状态和异常堆栈；
- 截图、OCR、修正、预测和终局均不写入文件；终局或退出立即清空本局内存；
- 静态模型文件是唯一允许持久化的助手资产。

安装仓库和助手：

```powershell
python -m pip install -e .
python -m pip install -e .\apps\ephemeral-assistant
```

从 `latest-model-v6-v2.0.0-beta.5` 预发布解压五个模型文件后运行：

```powershell
auction-vision-assistant `
  --models C:\path\to\latest-model-v6-v2.0.0-beta.5 `
  --treasures .\data\v1\core\treasures.csv
```

或者让程序按固定版本和 SHA-256 下载静态模型：

```powershell
auction-vision-assistant `
  --download-models `
  --models .\models\latest-model-v6-v2.0.0-beta.5 `
  --treasures .\data\v1\core\treasures.csv
```

程序启动前会严格校验三份 YOLO、v6 joblib 和 world-model-v2 共五个文件的
大小与 SHA-256。`joblib` 只能加载本项目固定哈希的 Release 文件，不应加载
不可信的同名替换文件。

完整自动滚动和视觉状态机可以先运行 v6 + world-model-v2，但地图行数会保留为
`automatic_scroll_estimate` 并显示警告；用户可在左侧画布和右侧事件列表快速
复核，也可把地图行数升级为人工精确确认。低置信度或未解析事件仍会阻止转义；
达到阈值且通过图鉴品质/尺寸一致性检查的自动完整身份直接作为 OCR 身份证据，
人工新增完整身份则必须在兼容图鉴列表中确认。这个 OCR 获取路径并不等价于私有
结构化输入，因此公开融合结果仍是 beta、未校准且固定 `actionable=false`。
条件重采样结果同样明确标为未校准，不会升级为自动出价依据。诊断日志只驻留
当前进程内存，本局重置时继续保留，便于复制完整问题现场。

旧视觉权重 Release 仍可独立验证：

```powershell
python scripts\verify_vision_release.py `
  --release C:\path\to\auction-assistant-vision-models-v1.0.0.zip
```

助手会在每轮出价前的视觉门控时间窗内自动滚动左侧地图并恢复顶部；没有其他
游戏控制能力。详细隐私契约见
[`apps/ephemeral-assistant/PRIVACY.md`](apps/ephemeral-assistant/PRIVACY.md)。

最新模型的脱敏、数值等价探针、验证状态与许可证边界见
[`models/LATEST_MODEL_CARD.md`](models/LATEST_MODEL_CARD.md)。发布构建脚本只读取
私有 bundle，输出不含抓包解析器、事件码本、协议物品映射、原始会话 ID 或本机
路径；私有主仓库不会被修改。

## 数据概况

公开数据版本为 `auction-moment-public-v1`，观察日期为 2026-07-27 至
2026-08-02，当前包含：

| 表 | 行数 | 含义 |
|---|---:|---|
| `sessions.csv` | 1,816 | 场次级质量、结算与汇总字段 |
| `rounds.csv` | 9,080 | 五轮结构化明细 |
| `opponent_bids.csv` | 21,632 | 匿名化对手报价 |
| `treasures.csv` | 58,068 | 终局藏品与视觉/结构化特征，不含名称和图片 |
| `event_catalog.csv` | 46 | 当前数据中观察到的事件目录 |
| `event_instances.csv` | 9,326 | 已解析事件实例 |
| `event_offers.csv` | 18,664 | 个人事件候选记录 |
| `player_rounds.jsonl` | 13,656 | 严格协议确认的匿名玩家-场次-轮次研究行 |

完整行数、字节数和 SHA-256 见
[`data/v1/manifest.json`](data/v1/manifest.json)。

## 已复现结果

### 分层世界模型

`auction_moment_research.train_world_model` 从公开 `treasures.csv` 重建 120 个
目录身份和 1,794 场可训练对局，并生成分层概率模型：

```text
总件数 → 彩色件数 → 非彩品质数量 → 品质内规格 → 具体身份
```

世界模型 JSON 不提交到 Git；生成代码、输入哈希和一次验证输出哈希保存在
[`models/MODEL_CARD.md`](models/MODEL_CARD.md) 与
[`results/world-model-training-manifest.json`](results/world-model-training-manifest.json)。
可直接加载的同一再生输出随视觉 Release 发布。

### 对手行为聚类

冻结训练集为 702 条完整五轮轨迹，时间外留出集为 321 条：

- 完整五轮轨迹只支持 2 个稳健描述簇，不代表永久玩家人格；
- 第二轮结束后的前缀簇将第 5 轮高相对报价 Brier 从 0.214614 降至 0.188679；
- 相比简单的一轮状态桶，额外改善仅 0.003759，95% 区间
  `[-0.006744, 0.014492]`，没有证明稳定胜出。

因此聚类仍是研究结果，不是当前轮出价特征。完整结果见
[`results/opponent-clustering-v1/STUDY.md`](results/opponent-clustering-v1/STUDY.md)。

## 安装与复现

建议使用 Python 3.12：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

若要严格使用本次验证版本：

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

运行测试和公开发布检查：

```bash
python -m unittest discover -s tests -v
python scripts/verify_public_release.py
```

重新训练世界模型：

```bash
python -m auction_moment_research.train_world_model \
  --input data/v1/core/treasures.csv \
  --output models/generated/world-model-v1.json \
  --manifest models/generated/manifest.json
```

重新生成聚类研究：

```bash
python -m auction_moment_research.opponent_clustering \
  --research-root data/v1/opponents \
  --output-root results/reproduced-opponent-clustering \
  --overwrite
```

## 匿名化与数据边界

- 会话 ID 和玩家键使用未公开密钥的 HMAC-SHA256 重新生成；
- 玩家昵称、自身账号、原始玩家 ID、精确时间、本机路径和源文件路径均已移除；
- 对手研究只保留日期，足以复现按时间切分；
- 结算、最终数量和身份属于赛后标签，不能伪装成决策时信息；
- 对局截图、视频和协议证据仍只保存在私有采集工程中；人工身份选择所需的
  120 张 80x80 图鉴缩略图是唯一公开的游戏图片子集；
- Release 权重不包含训练图片，但仍会经过检查点元数据净化，防止训练路径和
  原始会话标识被二进制文件间接带出。

详见 [`data/v1/DATA_CARD.md`](data/v1/DATA_CARD.md)。

## 结论报告

- [藏品生成规律](reports/generation-patterns-v1.md)
- [公共事件与个人事件刷新规律](reports/event-refresh-patterns-v1.md)
- [对手行为原型研究](reports/opponent-archetypes-v1.md)
- [对手行为无监督聚类](reports/opponent-clustering-v1.md)

所有结论均是当前冻结样本的经验结果，不是服务器源码证明。任何用于新版本、
新档位或决策系统的主张，都需要新的未消费时间块和预先登记的验证协议。

## 许可证

- 代码：Apache License 2.0，见 [`LICENSE`](LICENSE)。
- `apps/ephemeral-assistant/`：AGPL-3.0-only，见
  [`apps/ephemeral-assistant/LICENSE`](apps/ephemeral-assistant/LICENSE)。
- Ultralytics `.pt` 模型资产：AGPL-3.0；具体边界见
  [`models/VISION_MODEL_CARD.md`](models/VISION_MODEL_CARD.md) 和
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
- `data/v1/`、`reports/` 和 `results/`：CC BY 4.0，见
  [`DATA_LICENSE.md`](DATA_LICENSE.md)。
- 公开图鉴缩略图和第三方游戏名称、商标不受本仓库开源许可证授权；权利仍归
  各自权利人，详见 `THIRD_PARTY_NOTICES.md`。
