# Security and privacy

Please do not open a public issue containing private player data, raw captures,
credentials, local paths, or a method for reversing published pseudonyms.

The real-time assistant does not collect telemetry and has no session-output
path. It may persist only versioned static model assets selected by the user.
Model downloads are rejected unless both the declared byte count and SHA-256
match `models/vision-release-v1.json`.

Do not upload a raw Ultralytics checkpoint directly: training arguments can
contain absolute data paths and repository roots. Build public model assets
with `scripts/build_vision_release.py`, then independently run
`scripts/verify_vision_release.py`.

Until a private disclosure address is configured, report suspected privacy
issues through GitHub's private vulnerability reporting feature. Treat a leaked
export key, raw player identifier, nickname, exact private timestamp, or source
capture as sensitive.
