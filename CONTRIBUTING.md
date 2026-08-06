# Contributing

The project has been hibernating since 2026-08-06 because the event is offline.
The repository may be archived and does not promise issue or pull-request
response during hibernation. If the event returns, start from
`season-2026-archive-v1` and follow `RETURN_RUNBOOK.md` before changing any
real-time capability claim.

Changes should preserve the Git repository's text-only boundary and the
assistant's no-session-persistence contract.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=apps/ephemeral-assistant/src \
  python -m unittest discover -s apps/ephemeral-assistant/tests -v
python scripts/verify_public_release.py
```

Do not contribute screenshots, recordings, packet captures, decrypted game
files, player names, exact private timestamps, packet-derived models, or
automatic-bid execution code. Model weights belong only in a versioned GitHub
Release built by the repository sanitization script; never commit them to Git.
New empirical claims must identify their feature-time boundary, sample split,
uncertainty, and whether the result was selected on the same data.
No change may claim a complete real-time system until at least one formal full
match has run from lobby detection through all five rounds and match end without
post-result features or an unrecorded manual bypass.
