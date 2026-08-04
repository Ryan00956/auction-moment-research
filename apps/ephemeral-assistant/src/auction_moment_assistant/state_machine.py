from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class ScreenState(str, Enum):
    UNKNOWN = "unknown"
    LOBBY = "lobby"
    EVENT_SELECTION = "event_selection"
    ROUND_ACTIVE = "round_active"
    BID_ENTRY = "bid_entry"
    BID_CONFIRM = "bid_confirm"
    ROUND_RANKING = "round_ranking"
    FINAL_RESULT = "final_result"


@dataclass(frozen=True)
class StateFeatures:
    top_white: float
    center_white: float
    event_button_blue: float
    personal_event_button_blue: float
    round_start_banner_dark: float
    final_skip_button_dark: float
    final_skip_button_white: float
    keypad_dark: float
    bid_button_blue: float
    left_saturation: float
    left_white: float


@dataclass(frozen=True)
class WatchAction:
    kind: str
    state: ScreenState
    round_number: int


ROIS = {
    "top": (300, 55, 980, 165),
    "center": (300, 170, 980, 535),
    "event_button": (840, 375, 1275, 480),
    "personal_event_button": (735, 470, 1010, 535),
    "round_start_banner": (190, 290, 1090, 355),
    "final_skip_button": (485, 15, 665, 80),
    "keypad": (20, 385, 730, 690),
    "bid_button": (835, 585, 1270, 700),
    "left": (35, 80, 655, 690),
}


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return array[top:bottom, left:right]


def _white_fraction(rgb: np.ndarray) -> float:
    channel_min = rgb.min(axis=2)
    channel_span = rgb.max(axis=2) - channel_min
    return float(((channel_min > 210) & (channel_span < 35)).mean())


def _dark_fraction(rgb: np.ndarray) -> float:
    return float((rgb.mean(axis=2) < 175).mean())


def _blue_fraction(hsv: np.ndarray) -> float:
    return float(
        (
            (hsv[:, :, 0] > 90)
            & (hsv[:, :, 0] < 130)
            & (hsv[:, :, 1] > 90)
            & (hsv[:, :, 2] > 70)
        ).mean()
    )


def extract_features(frame_rgb: np.ndarray) -> StateFeatures:
    rgb = np.asarray(frame_rgb, dtype=np.uint8)
    if rgb.shape[:2] != (720, 1280):
        raise ValueError(
            f"状态机输入尺寸为 {rgb.shape[1]}x{rgb.shape[0]}，期望 1280x720"
        )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    final_skip = _crop(rgb, ROIS["final_skip_button"])
    left_hsv = _crop(hsv, ROIS["left"])
    return StateFeatures(
        top_white=_white_fraction(_crop(rgb, ROIS["top"])),
        center_white=_white_fraction(_crop(rgb, ROIS["center"])),
        event_button_blue=_blue_fraction(_crop(hsv, ROIS["event_button"])),
        personal_event_button_blue=_blue_fraction(
            _crop(hsv, ROIS["personal_event_button"])
        ),
        round_start_banner_dark=_dark_fraction(
            _crop(rgb, ROIS["round_start_banner"])
        ),
        final_skip_button_dark=_dark_fraction(final_skip),
        final_skip_button_white=_white_fraction(final_skip),
        keypad_dark=_dark_fraction(_crop(rgb, ROIS["keypad"])),
        bid_button_blue=_blue_fraction(_crop(hsv, ROIS["bid_button"])),
        left_saturation=float(left_hsv[:, :, 1].mean()),
        left_white=_white_fraction(_crop(rgb, ROIS["left"])),
    )


def has_final_result_signature(features: StateFeatures) -> bool:
    return bool(
        features.personal_event_button_blue < 0.20
        and features.final_skip_button_dark > 0.45
        and features.final_skip_button_white < 0.08
        and features.keypad_dark < 0.30
        and features.bid_button_blue > 0.50
    )


def classify_state(features: StateFeatures) -> ScreenState:
    # Order is part of the calibrated contract: modal screens cover the
    # normal round controls and therefore must be checked first.
    if features.top_white < 0.12 and features.left_saturation > 65:
        return ScreenState.LOBBY
    if has_final_result_signature(features):
        return ScreenState.FINAL_RESULT
    if features.top_white > 0.68 and features.center_white > 0.40:
        return ScreenState.ROUND_RANKING
    if features.top_white < 0.20 and features.center_white > 0.62:
        return ScreenState.BID_CONFIRM
    if features.event_button_blue > 0.40 and features.bid_button_blue < 0.15:
        return ScreenState.EVENT_SELECTION
    if features.keypad_dark > 0.35 and features.top_white < 0.65:
        return ScreenState.BID_ENTRY
    if (
        features.personal_event_button_blue > 0.40
        and features.bid_button_blue > 0.50
        and features.keypad_dark < 0.30
    ):
        return ScreenState.ROUND_ACTIVE
    if features.top_white > 0.22 and features.left_white > 0.24:
        return ScreenState.ROUND_ACTIVE
    return ScreenState.UNKNOWN


class DebouncedState:
    def __init__(self, required_frames: int = 2) -> None:
        self.required_frames = max(1, int(required_frames))
        self.candidate = ScreenState.UNKNOWN
        self.candidate_count = 0
        self.stable = ScreenState.UNKNOWN

    def update(self, observed: ScreenState) -> ScreenState:
        if observed == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = observed
            self.candidate_count = 1
        required = 1 if observed == ScreenState.BID_CONFIRM else self.required_frames
        if self.candidate_count >= required:
            self.stable = observed
        return self.stable


class RoundWatchStateMachine:
    """Pure session/round watcher. It never performs device input itself."""

    def __init__(self, *, fallback_active_frames: int = 5) -> None:
        self.fallback_active_frames = max(2, int(fallback_active_frames))
        self.state = ScreenState.UNKNOWN
        self.round_number = 1
        self.session_active = False
        self._ranking_seen = False
        self._banner_seen = False
        self._active_without_banner = 0
        self._scan_pending = False
        self._scanned_rounds: set[int] = set()

    def _begin_session(self) -> None:
        self.session_active = True
        self.round_number = 1
        self._ranking_seen = False
        self._banner_seen = False
        self._active_without_banner = 0
        self._scan_pending = False
        self._scanned_rounds.clear()

    def set_round(self, round_number: int) -> None:
        self.round_number = max(1, min(5, int(round_number)))
        self._banner_seen = False
        self._active_without_banner = 0
        self._scan_pending = False

    def observe(
        self, state: ScreenState, features: StateFeatures
    ) -> tuple[WatchAction, ...]:
        actions: list[WatchAction] = []
        previous = self.state
        self.state = state

        if not self.session_active and state not in {
            ScreenState.UNKNOWN,
            ScreenState.LOBBY,
            ScreenState.FINAL_RESULT,
        }:
            self._begin_session()
            actions.append(WatchAction("session_started", state, self.round_number))

        if state != previous:
            actions.append(WatchAction("state_changed", state, self.round_number))

        if state == ScreenState.FINAL_RESULT:
            if self.session_active:
                self.session_active = False
                self._scan_pending = False
                actions.append(WatchAction("session_ended", state, self.round_number))
            return tuple(actions)

        if state == ScreenState.LOBBY:
            self._active_without_banner = 0
            if self.session_active:
                self.session_active = False
                self._scan_pending = False
                actions.append(WatchAction("session_ended", state, self.round_number))
            return tuple(actions)

        if not self.session_active:
            return tuple(actions)

        if state == ScreenState.ROUND_RANKING:
            self._ranking_seen = True
            self._active_without_banner = 0
        elif state == ScreenState.EVENT_SELECTION and self._ranking_seen:
            self.round_number = min(5, self.round_number + 1)
            self._ranking_seen = False
            self._banner_seen = False
            self._active_without_banner = 0
            self._scan_pending = False
            actions.append(WatchAction("round_changed", state, self.round_number))
        elif state == ScreenState.ROUND_ACTIVE:
            if features.round_start_banner_dark >= 0.55:
                self._banner_seen = True
                self._active_without_banner = 0
            else:
                self._active_without_banner += 1
            ready = (
                self._banner_seen and features.round_start_banner_dark <= 0.40
            ) or self._active_without_banner >= self.fallback_active_frames
            if (
                ready
                and self.round_number not in self._scanned_rounds
                and not self._scan_pending
            ):
                self._scan_pending = True
                actions.append(WatchAction("scan_round", state, self.round_number))
        else:
            self._active_without_banner = 0
        return tuple(actions)

    def mark_scan_result(self, round_number: int, success: bool) -> None:
        if int(round_number) != self.round_number:
            return
        self._scan_pending = False
        if success:
            self._scanned_rounds.add(self.round_number)
        else:
            # Require a fresh stable active window before retrying.
            self._active_without_banner = 0
