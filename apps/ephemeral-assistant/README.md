# Auction Moment Ephemeral Assistant

这是公开研究仓库中的独立半自动实时视觉助手。状态机常驻读取 `1280×720`
画面：大厅待机，检测开局，在每轮出价前自动归顶并分段扫描地图，识别完成后
恢复顶部、输出预测并待机。RapidOCR 和 Release 中的三份 YOLO 权重负责中间
过程识别，公开 OCR 转义层调用最新冻结的 v6 + world-model-v2。

运行时不包含抓包、协议解析、任意点击、键盘输入或自动出价；自动输入能力只有
两种固定的左侧地图滚动手势。不保存截图、OCR、人工修改、预测或最终结算。
静态模型文件是唯一允许持久化的运行资产。最新 v6 的
候选 ID 是 `protocol-round-residual-v6-72a08074c3d8af81`。

左侧是可直接操作的地图建模画布：空白处拖拽新增，拖动已有对象移动，点选后
修改品质/规格/身份，右键删除。右侧按轮列出公共/个人事件，选中即可修正。
修改只驻留内存并立即重算。

OCR 转义层不会伪造抓包字段：事件必须达到 OCR 阈值或经人工修正；自动扫描的
地图行数明确标为估算，用户可复核后升级为人工精确确认。YOLO 给出的身份只作为
候选，必须人工确认后才作为精确身份输入 v6。OCR 路径与原始结构化输入不等价，
区间仍未校准，所有结果固定为 `actionable=false`。

本子项目因使用 Ultralytics 按 AGPL-3.0-only 发布；仓库根目录的离线研究包仍为
Apache-2.0，数据、报告和结果仍为 CC BY 4.0。

开发安装：

```powershell
python -m pip install -e .
python -m pip install -e .\apps\ephemeral-assistant
auction-vision-assistant --models C:\path\to\latest-model-v6-v2.0.0-beta.2
```

启动后不需要再点击“开始识别”：程序会自动进入大厅监视。需要纠错时直接操作
左侧地图或右侧事件；“立即重扫本轮”只在轮次主界面使用。

若模型尚未下载，可直接使用 `--download-models`；也可以通过
`--model-base-url` 指向兼容镜像。下载器只写入静态模型目录，不写入任何
对局数据。

程序只加载清单中固定大小和 SHA-256 的 `joblib`。Joblib/pickle 文件可能在
加载时执行代码，请勿把同名的第三方文件放入模型目录。
