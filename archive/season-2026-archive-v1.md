# 2026 赛季公开归档快照

`season-2026-archive-v1` 是“竞拍时刻”活动下线后的权威公开快照。它把此前分散在 Git 标签、GitHub Release 和仓库数据目录中的公开材料合并为一个可校验的恢复入口。

## 这个快照代表什么

- 研究数据、分析结论和冻结模型在 2026-08-06 的公开状态；
- 一个可供审阅和未来重新验证的 OCR/视觉助手原型；
- 项目进入 `HIBERNATING` 状态的明确边界；
- 不再变化的公开版本和 SHA-256 清单。

它不代表实时推理已经在正式活动环境完成一整局端到端跑通，也不代表模型完成前瞻盲测、生产启用或收益验证。

## Release 资产

- `auction-moment-public-archive-v1.zip`：公开数据、报告、结果、数据卡、模型卡和状态文档；
- `auction_moment_research-0.1.0-py3-none-any.whl`：研究代码；
- `auction_moment_assistant-0.2.0b6-py3-none-any.whl`：实验性助手；
- 三个 YOLO 模型、v6 价值模型和 world-model-v2；
- `archive-release-manifest.json`：包含源提交、每个资产的角色、字节数和 SHA-256；
- `SHA256SUMS.txt`：可独立校验的总哈希清单；
- `requirements-runtime-win-py312.txt`：最后一次已知可用的 Windows Python 3.12 依赖版本；
- `install_archive.ps1`：只负责校验和恢复，不自动启动助手或连接设备；
- `STATUS.md` 与 `RETURN_RUNBOOK.md`：能力边界和返场门槛。

两个 wheel 直接复用已经独立验证的 beta.6 二进制及其原始 SHA-256，不以相同版本号重新构建不同内容。助手 wheel 中的 120 张图鉴缩略图不在项目许可授权范围内，权利边界见 `THIRD_PARTY_NOTICES.md`。

## 校验

下载同一 Release 的全部资产后：

```powershell
Get-Content .\SHA256SUMS.txt | ForEach-Object {
  if ($_ -notmatch '^([0-9a-f]{64})  (.+)$') { throw "bad checksum line" }
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Matches[2]).Hash.ToLowerInvariant() -ne $Matches[1]) {
    throw "checksum mismatch: $($Matches[2])"
  }
}
```

也可以使用仓库中的验证器：

```powershell
python .\scripts\verify_archive_release.py --bundle-dir C:\path\to\downloaded-assets
```

恢复脚本默认调用 `py -3.12`；如果 Windows 启动器没有注册解释器，可传入 `-PythonExecutable C:\path\to\python.exe`。

返场时不要从旧 beta 标签直接声明恢复可用；应从本归档标签创建新分支，并完整执行 `RETURN_RUNBOOK.md`。
