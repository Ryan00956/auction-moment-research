# Privacy contract

The assistant is designed around an application-level no-persistence boundary.

## Volatile session data

The following values exist only in process memory and are never intentionally
written to disk or transmitted over a network:

- screenshots and normalized map viewports;
- OCR text, confidence values, and parsed event semantics;
- visual detections and identity Top-k candidates;
- manual corrections and undo/revision state;
- compatible-world predictions;
- final-result frames or settlement values.

The assistant overwrites the frame buffer it owns before releasing it. Third-
party native libraries, GPU drivers, operating-system paging, emulator logs,
and crash dumps are outside this application-level guarantee.

## Persistent static assets

The executable environment, public text dataset, and versioned model files are
normal static inputs and may be stored locally. The optional downloader writes
only files declared in `model-release.json`, verifies their byte counts and
SHA-256 values, and never uploads data.

## Device access

The ADB frame source invokes only `adb [-s SERIAL] exec-out screencap -p` after
device discovery. It does not issue taps, swipes, key events, packet capture,
port forwarding, process injection, protocol decoding, or settlement export.

## Fail-closed behavior

Missing/unparsed events, unconfirmed map completeness, low public-world support,
or contradictory constraints remain provisional or blocked. Human edits can
repair observations but cannot toggle an estimate into an automatic action;
all published assistant results have `actionable=false`.
