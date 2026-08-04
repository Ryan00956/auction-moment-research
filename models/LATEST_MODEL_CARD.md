# Latest v6 + world-model-v2 model card

This release contains the newest frozen value-model candidate and generative
world model used by the project at the time of publication:

- `latest-value-model-v6.joblib`: candidate
  `protocol-round-residual-v6-72a08074c3d8af81`;
- `world-model-v2-S.json`: the formula-driven S-tier generative prior.

The v6 estimator parameters, numeric reference state, and model topology are
preserved. Raw session identifiers, private filesystem paths, packet parsers,
event codebooks, protocol item mappings, per-session captures, player identity,
settlement records, and automatic bidding code are not included. The assistant
wheel does include the separately documented 120 public 80x80 catalog
thumbnails used for manual identity selection. The release builder compares
original and sanitized predictions on deterministic probes.

## OCR adapter boundary

The public assistant uses `ocr-visible-pre-bid-adapter-v1`. It accepts parsed
events above the configured OCR threshold or human-confirmed edits. Automatic
visual pre-bid proof, full scroll scans, and estimated map height may enter the
adapter with explicit warnings; they are never relabeled as human-confirmed or
exact. It never invents packet indexes. High-confidence, catalog-compatible
automatic identity OCR is accepted by default, while a manually assigned exact
identity still requires explicit confirmation in the catalog selector.

This changes the evidence acquisition path: OCR-adapted inference is not
validated as equivalent to the private structured-input path. Outputs are
uncalibrated research estimates with `actionable=false`; the assistant never
submits a bid.

If the initial finite world population is eliminated, world-model-v2 performs
a deterministic, cached conditional rejection resample of up to 32,768 worlds.
A recovered result is explicitly provisional and uncalibrated. Exhaustion after
resampling falls back to the frozen v6 interval and manual-only conservative bid
advice; it is not presented as a proof that the conditions are impossible.

## Validation status

The original v6 evaluation was retrospective, not a fresh blind promotion.
The candidate remains frozen for future blind validation and is not a
production-enabled model. Releasing it makes the exact frozen candidate usable
and developable; it does not upgrade its validation status.

## Loading safety

Joblib/pickle formats can execute code while loading. The assistant only loads
the release filename after size and SHA-256 verification. Do not load an
untrusted replacement file merely because it has the same filename.

## License

The v6 and world-model-v2 artifacts are released under Apache License 2.0. The
packaged assistant source, including the compatibility runtime needed to load
the joblib, and the YOLO assets remain under their documented AGPL-3.0 terms.
