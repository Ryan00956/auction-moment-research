from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .capture import FrameSource
from .observations import ObservationStore
from .ocr import FixedLayoutOCR, OcrCapture
from .predictor import EmpiricalWorldPredictor, PredictionResult
from .vision import VisionRecognizer


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return array[top:bottom, left:right]


def _white_fraction(rgb: np.ndarray) -> float:
    channel_min = rgb.min(axis=2)
    channel_span = rgb.max(axis=2) - channel_min
    return float(((channel_min > 210) & (channel_span < 35)).mean())


def _dark_fraction(rgb: np.ndarray) -> float:
    return float((rgb.mean(axis=2) < 175).mean())


def _blue_fraction(rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(
        (
            (hsv[:, :, 0] > 90)
            & (hsv[:, :, 0] < 130)
            & (hsv[:, :, 1] > 90)
            & (hsv[:, :, 2] > 70)
        ).mean()
    )


def is_final_result(frame_rgb: np.ndarray) -> bool:
    array = np.asarray(frame_rgb, dtype=np.uint8)
    if array.shape[:2] != (720, 1280):
        return False
    personal_event = _blue_fraction(_crop(array, (735, 470, 1010, 535)))
    final_skip = _crop(array, (485, 15, 665, 80))
    keypad = _crop(array, (20, 385, 730, 690))
    action_button = _blue_fraction(_crop(array, (835, 585, 1270, 700)))
    return bool(
        personal_event < 0.20
        and _dark_fraction(final_skip) > 0.45
        and _white_fraction(final_skip) < 0.08
        and _dark_fraction(keypad) < 0.30
        and action_button > 0.50
    )


@dataclass(frozen=True)
class CaptureOutcome:
    revision: int
    final_detected: bool
    ocr: OcrCapture | None
    map_item_count: int
    errors: tuple[str, ...]


class CapturePipeline:
    def __init__(
        self,
        *,
        source: FrameSource,
        store: ObservationStore,
        ocr: FixedLayoutOCR | None,
        vision: VisionRecognizer | None,
    ) -> None:
        self.source = source
        self.store = store
        self.ocr = ocr
        self.vision = vision

    def capture_once(
        self, *, expected_round: int, offset_rows: int
    ) -> CaptureOutcome:
        frame = self.source.capture()
        self.store.set_frame(frame)
        if is_final_result(frame):
            revision = self.store.reset()
            return CaptureOutcome(
                revision=revision,
                final_detected=True,
                ocr=None,
                map_item_count=0,
                errors=(),
            )
        errors = []
        ocr_capture = None
        if self.ocr is not None:
            try:
                ocr_capture = self.ocr.recognize(
                    frame, expected_round=int(expected_round)
                )
            except Exception as exc:
                errors.append(f"OCR: {exc}")
        map_items = ()
        if self.vision is not None:
            try:
                map_items = self.vision.recognize(
                    frame, offset_rows=int(offset_rows)
                )
            except Exception as exc:
                errors.append(f"vision: {exc}")
        revision = self.store.apply_capture(
            round_number=(
                ocr_capture.round_number
                if ocr_capture is not None
                else int(expected_round)
            ),
            events=ocr_capture.events if ocr_capture is not None else (),
            bankroll=(
                ocr_capture.bankroll if ocr_capture is not None else None
            ),
            map_items=map_items,
        )
        return CaptureOutcome(
            revision=revision,
            final_detected=False,
            ocr=ocr_capture,
            map_item_count=len(map_items),
            errors=tuple(errors),
        )


class InferenceCoordinator:
    def __init__(self, predictor: EmpiricalWorldPredictor) -> None:
        self.predictor = predictor
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="auction-predictor"
        )

    def submit(self, store: ObservationStore) -> Future[PredictionResult]:
        snapshot = store.snapshot()
        return self._executor.submit(self.predictor.predict, snapshot)

    @staticmethod
    def current(
        store: ObservationStore, result: PredictionResult
    ) -> bool:
        return int(result.revision) == int(store.revision)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
