from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .observations import EventObservation
from .semantics import event_semantics


CURRENT_PERSONAL_TEXT_CANDIDATES = (
    ((680, 138, 1000, 174),),
    ((680, 138, 1000, 171), (680, 168, 1000, 199)),
    ((675, 134, 1005, 179),),
)
CURRENT_PUBLIC_TEXT_CANDIDATES = (
    ((680, 210, 1000, 246),),
    ((680, 210, 1000, 243), (680, 240, 1000, 271)),
    ((680, 235, 1000, 270),),
    ((680, 235, 1000, 268), (680, 265, 1000, 296)),
    ((675, 206, 1005, 251),),
)
BANKROLL_TEXT_CANDIDATES = (
    ((845, 558, 1108, 598),),
    ((838, 553, 1115, 602),),
)
ROUND_HEADER_TEXT_CANDIDATES = (
    ((35, 18, 290, 62),),
    ((20, 10, 320, 75),),
)


class OcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrCapture:
    round_number: int
    round_verified: bool
    events: tuple[EventObservation, ...]
    bankroll: int | None
    timing_ms: float


def bankroll_from_text(text: str) -> int | None:
    digits = "".join(re.findall(r"\d", str(text)))
    return int(digits) if len(digits) >= 4 else None


def clean_event_text(text: str) -> str:
    return re.sub(
        r"^第[一二三四五1-5]轮(?:公共|个人)事件[·#：:]?",
        "",
        str(text).strip(),
    ).strip()


def round_number_from_header(text: str) -> int | None:
    match = re.search(r"第\s*([一二三四五1-5])\s*轮", str(text))
    if match is None:
        return None
    token = match.group(1)
    return {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
    }.get(token, int(token) if token.isdigit() else None)


class FixedLayoutOCR:
    def __init__(
        self,
        engine=None,
        *,
        minimum_confidence: float = 0.85,
    ) -> None:
        if engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OcrError("请安装 rapidocr-onnxruntime==1.4.4") from exc
            engine = RapidOCR()
        self.engine = engine
        self.minimum_confidence = float(minimum_confidence)

    def _recognize_candidates(
        self,
        image: np.ndarray,
        candidates: Sequence[Sequence[tuple[int, int, int, int]]],
        *,
        semantic: bool,
    ) -> dict:
        attempts = []
        cache = {}
        for boxes in candidates:
            pieces = []
            for box in boxes:
                if box not in cache:
                    left, top, right, bottom = box
                    crop = image[top:bottom, left:right]
                    result, _timing = self.engine(
                        crop,
                        use_det=False,
                        use_cls=False,
                        use_rec=True,
                    )
                    recognized = [
                        (str(raw[0]), float(raw[1]))
                        for raw in (result or [])
                        if len(raw) >= 2
                    ]
                    cache[box] = max(
                        recognized,
                        key=lambda value: value[1],
                        default=("", 0.0),
                    )
                pieces.append(cache[box])
            nonempty = [piece for piece in pieces if piece[0]]
            text = "".join(piece[0] for piece in nonempty)
            confidence = (
                sum(piece[1] for piece in nonempty) / len(nonempty)
                if nonempty
                else 0.0
            )
            parsed = (
                event_semantics(clean_event_text(text)).get("parsed")
                if semantic
                else bankroll_from_text(text) is not None
            )
            attempt = {
                "text": text,
                "confidence": float(confidence),
                "parsed": bool(parsed),
            }
            attempts.append(attempt)
            if parsed and confidence >= self.minimum_confidence:
                break
        return max(
            attempts,
            key=lambda value: (
                int(value["parsed"]), float(value["confidence"])
            ),
        )

    def recognize(
        self,
        image: np.ndarray,
        *,
        expected_round: int | None = None,
    ) -> OcrCapture:
        array = np.asarray(image, dtype=np.uint8)
        if array.shape[:2] != (720, 1280):
            raise OcrError(
                f"OCR 输入尺寸为 {array.shape[1]}x{array.shape[0]}，期望 1280x720"
            )
        started = time.perf_counter()
        header = self._recognize_candidates(
            array, ROUND_HEADER_TEXT_CANDIDATES, semantic=False
        )
        observed_round = round_number_from_header(header["text"])
        round_number = int(expected_round or observed_round or 1)
        round_verified = bool(
            observed_round == round_number
            and float(header["confidence"]) >= self.minimum_confidence
        )
        personal = self._recognize_candidates(
            array, CURRENT_PERSONAL_TEXT_CANDIDATES, semantic=True
        )
        public = self._recognize_candidates(
            array, CURRENT_PUBLIC_TEXT_CANDIDATES, semantic=True
        )
        bankroll = self._recognize_candidates(
            array, BANKROLL_TEXT_CANDIDATES, semantic=False
        )
        events = []
        for kind, value in (("public", public), ("personal", personal)):
            text = clean_event_text(value["text"])
            events.append(
                EventObservation(
                    round_number=round_number,
                    kind=kind,
                    text=text,
                    confidence=float(value["confidence"]),
                    semantics=event_semantics(text),
                )
            )
        return OcrCapture(
            round_number=round_number,
            round_verified=round_verified,
            events=tuple(events),
            bankroll=bankroll_from_text(bankroll["text"]),
            timing_ms=round((time.perf_counter() - started) * 1000, 3),
        )
