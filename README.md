# Auction Moment Research

这是一个从私有采集工程中独立导出的公开研究仓库，包含：

- 匿名化、纯文本、带哈希清单的数据集；
- 可从公开数据重新生成的概率世界模型和对手行为聚类代码；
- 冻结样本上的分析结论、评估指标和不确定性边界；
- 数据匿名化、模型训练和公开发布检查工具。

本仓库是**离线研究项目**。它不包含图片、视频、抓包、解密脚本、游戏二进制、
模型权重、自动点击或自动出价执行层，也不授权将研究结果直接用于实时建议。

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

模型 JSON 不提交到 Git；生成代码、输入哈希和一次验证输出哈希保存在
[`models/MODEL_CARD.md`](models/MODEL_CARD.md) 与
[`results/world-model-training-manifest.json`](results/world-model-training-manifest.json)。

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
- 原始图片、视频和协议证据仍只保存在私有采集工程中。

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
- `data/v1/`、`reports/` 和 `results/`：CC BY 4.0，见
  [`DATA_LICENSE.md`](DATA_LICENSE.md)。
- 第三方游戏名称、商标和未包含的原始资产不因本仓库获得任何授权。
