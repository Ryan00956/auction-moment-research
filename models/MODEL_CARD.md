# 模型卡

## 分层世界模型

实现：`auction_moment_research.world_model.ProbabilisticWorldModel`

训练入口：

```bash
python -m auction_moment_research.train_world_model \
  --input data/v1/core/treasures.csv \
  --output models/generated/world-model-v1.json \
  --manifest models/generated/manifest.json
```

本次验证输入为 120 个目录身份、1,794 场 S 档对局。固定公开输入生成的验证
输出 SHA-256 为：

```text
d7ca7caaca5684264dc021da5a255bb8bfab2a2945c0d8f390d57a8958e8b428
```

模型文件是可再生输出，因此 `models/generated/` 被 Git 忽略。该哈希用于发现
代码、依赖或数据漂移，不表示模型已经通过前瞻生产验证。

## 对手聚类模型

实现：`auction_moment_research.opponent_clustering`

- 完整轨迹簇是赛后描述标签；
- 两轮前缀簇只允许在第 2 轮结算后研究第 5 轮；
- 玩家身份和事件码不进入聚类特征；
- 结果未接入建议、自动出价或当前轮模型。

## 不包含的模型

视觉检测和分类权重没有发布，因为公开数据不含其图片训练证据。仓库也不提交
`.pt`、`.onnx`、`.joblib` 或其他不可审计的二进制模型文件。
