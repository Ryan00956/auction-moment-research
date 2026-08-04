# Auction Moment Ephemeral Assistant

这是公开研究仓库中的独立实时视觉助手。它只读取当前 `1280×720` 画面，使用
RapidOCR 和 Release 中的三份 YOLO 权重识别中间过程，允许用户实时修改错误，
并基于匿名公开数据输出研究估计。

运行时不包含抓包、协议解析、自动点击或自动出价；不保存截图、OCR、人工修改、
预测或最终结算。静态模型文件是唯一允许持久化的运行资产。

本子项目因使用 Ultralytics 按 AGPL-3.0-only 发布；仓库根目录的离线研究包仍为
Apache-2.0，数据、报告和结果仍为 CC BY 4.0。

开发安装：

```powershell
python -m pip install -e .
python -m pip install -e .\apps\ephemeral-assistant
auction-vision-assistant --models C:\path\to\vision-models-v1.0.0
```

若模型尚未下载，可直接使用 `--download-models`；也可以通过
`--model-base-url` 指向兼容镜像。下载器只写入静态模型目录，不写入任何
对局数据。
