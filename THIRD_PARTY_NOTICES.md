# Third-party notices

## Ultralytics

The optional real-time vision assistant loads models with the Ultralytics
Python package. Ultralytics 8.4.104 declares the AGPL-3.0 license. The release
bundle includes the license text supplied by the installed distribution.

Official project and licensing information:

- https://github.com/ultralytics/ultralytics
- https://www.ultralytics.com/license

## RapidOCR

The assistant uses `rapidocr-onnxruntime` 1.4.4 for text recognition. The
package declares Apache-2.0. RapidOCR states that OCR model copyright is held
by Baidu; users should retain the notices shipped by that dependency.

Official project:

- https://github.com/RapidAI/RapidOCR

## PyTorch and ONNX Runtime

The assistant may install PyTorch, Torchvision, and ONNX Runtime as separate
runtime dependencies. Those packages are not redistributed in this repository
or in the model-only Release bundle. Their own license files remain
authoritative.
## Treasure catalog thumbnails

The 80x80 treasure thumbnails under
`apps/ephemeral-assistant/src/auction_moment_assistant/catalog_previews/` are
derived from in-game catalog artwork and are included for identification and
interoperability in the manual OCR correction interface. They are not licensed
under Apache-2.0, AGPL-3.0, or CC BY 4.0 by this repository. All rights in the
underlying game artwork, names, and trademarks remain with their respective
owners.
