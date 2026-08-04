# Latest v6 + world-model-v2 model card

This release contains the newest frozen value-model candidate and generative
world model used by the project at the time of publication:

- `latest-value-model-v6.joblib`: candidate
  `protocol-round-residual-v6-72a08074c3d8af81`;
- `world-model-v2-S.json`: the formula-driven S-tier generative prior.

The v6 estimator parameters, numeric reference state, and model topology are
preserved. Raw session identifiers, private filesystem paths, packet parsers,
event codebooks, protocol item mappings, captures, images, player identity,
settlement records, and automatic bidding code are not included. The release
builder compares original and sanitized predictions on deterministic probes.

## OCR adapter boundary

The public assistant uses `ocr-visible-pre-bid-adapter-v1`. It accepts parsed
events above the configured OCR threshold or human-confirmed edits. It requires
the user to confirm the pre-bid moment, full visible-map review, and exact map
height. It never invents packet indexes. Vision identity candidates only become
exact model identities after a manual confirmation.

This changes the evidence acquisition path: OCR-adapted inference is not
validated as equivalent to the private structured-input path. Outputs are
uncalibrated research estimates with `actionable=false`; the assistant never
submits a bid.

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
