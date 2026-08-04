# Auction Moment Ephemeral Assistant

这是公开研究仓库中的独立实时视觉助手。它只读取当前 `1280×720` 画面，使用
RapidOCR 和 Release 中的三份 YOLO 权重识别中间过程，允许用户实时修改错误，
并通过公开的 OCR 转义层调用最新冻结的 v6 + world-model-v2，输出研究估计。

运行时不包含抓包、协议解析、自动点击或自动出价；不保存截图、OCR、人工修改、
预测或最终结算。静态模型文件是唯一允许持久化的运行资产。最新 v6 的
候选 ID 是 `protocol-round-residual-v6-72a08074c3d8af81`。

OCR 转义层不会伪造抓包字段：事件必须达到 OCR 阈值或经人工修正；用户还需
确认当前画面处于出价前、可见地图已复核，以及地图底部和总行数。YOLO 给出的
身份只作为候选，必须人工确认后才作为精确身份输入 v6。OCR 路径与原始结构化
输入不等价，区间仍未校准，所有结果固定为 `actionable=false`。

本子项目因使用 Ultralytics 按 AGPL-3.0-only 发布；仓库根目录的离线研究包仍为
Apache-2.0，数据、报告和结果仍为 CC BY 4.0。

开发安装：

```powershell
python -m pip install -e .
python -m pip install -e .\apps\ephemeral-assistant
auction-vision-assistant --models C:\path\to\latest-model-v6-v2.0.0-beta.1
```

若模型尚未下载，可直接使用 `--download-models`；也可以通过
`--model-base-url` 指向兼容镜像。下载器只写入静态模型目录，不写入任何
对局数据。

程序只加载清单中固定大小和 SHA-256 的 `joblib`。Joblib/pickle 文件可能在
加载时执行代码，请勿把同名的第三方文件放入模型目录。
