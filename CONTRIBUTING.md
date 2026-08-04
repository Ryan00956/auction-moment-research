# Contributing

Changes should preserve the repository's research-only and text-only boundary.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
python scripts/verify_public_release.py
```

Do not contribute screenshots, recordings, packet captures, decrypted game
files, player names, exact private timestamps, model weights, or automatic-bid
execution code. New empirical claims must identify their feature-time boundary,
sample split, uncertainty, and whether the result was selected on the same data.
