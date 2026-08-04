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
- the scrollable diagnostic log, including OCR/map snapshots and tracebacks;
- final-result frames or settlement values.

The assistant overwrites the frame buffer it owns before releasing it. Third-
party native libraries, GPU drivers, operating-system paging, emulator logs,
and crash dumps are outside this application-level guarantee.

## Persistent static assets

The executable environment, public text dataset, and versioned model files are
normal static inputs and may be stored locally. The public package also includes
120 deliberately published 80x80 treasure-catalog thumbnails for manual identity
selection; these are static application assets, not captured session frames.
The optional downloader writes
only files declared in `model-release.json`, verifies their byte counts and
SHA-256 values, and never uploads data.

The latest v6 joblib and world-model-v2 are static assets. The release builder
replaces raw session identifiers and private paths and verifies numeric
equivalence before publication. Packet parsers, protocol codebooks, automatic
bidding logic, per-session screenshots, and settlement records are not included.

## Device access

After device discovery, the ADB source may invoke only:

- `adb [-s SERIAL] exec-out screencap -p`;
- `adb [-s SERIAL] shell input swipe 340 220 340 630 420` to move the left
  map toward its top;
- `adb [-s SERIAL] shell input swipe 340 560 340 320 650` to advance one map
  viewport.

The public input API rejects every other gesture. It has no tap, key, text,
bid, packet-capture, port-forwarding, process-injection, protocol-decoding, or
settlement-export operation. Scanned viewports are held only as short-lived
arrays while recognition runs and are discarded after the map is restored.

## Fail-closed behavior

Missing/unparsed events, an unproven pre-bid moment, an incomplete scan, low
public-world support, or contradictory constraints remain provisional or
blocked. A complete automatic scroll may supply an explicitly estimated map
height to v6, but it is never relabeled as exact; the UI retains warnings until
human review. Human edits can repair observations but cannot toggle an
estimate into an automatic action; all published assistant results have
`actionable=false`.
