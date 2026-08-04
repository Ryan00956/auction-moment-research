from __future__ import annotations

from auction_moment_assistant.ui import (
    AssistantWindow,
    INITIAL_SASH_RETRY_MS,
    preferred_main_sash,
)


class _FakeRoot:
    def __init__(self) -> None:
        self.callbacks: list[tuple[int, object]] = []

    def update_idletasks(self) -> None:
        return None

    def after(self, delay: int, callback) -> None:
        self.callbacks.append((delay, callback))


class _FakePane:
    def __init__(self, width: int) -> None:
        self.width = width
        self.sash = 0

    def winfo_width(self) -> int:
        return self.width

    def sashpos(self, _index: int, value: int | None = None) -> int:
        if value is not None:
            self.sash = value
        return self.sash


def test_preferred_main_sash_waits_for_real_geometry() -> None:
    assert preferred_main_sash(1) is None
    assert preferred_main_sash(899) is None
    assert preferred_main_sash(1424) == 882


def test_initial_sash_retries_after_one_pixel_layout() -> None:
    window = object.__new__(AssistantWindow)
    window.root = _FakeRoot()
    window.main_pane = _FakePane(width=1)
    window._initial_sash_attempts = 0
    window._initial_sash_applied = False

    window._set_initial_sashes()

    assert window.main_pane.sash == 0
    assert len(window.root.callbacks) == 1
    delay, callback = window.root.callbacks.pop()
    assert delay == INITIAL_SASH_RETRY_MS

    window.main_pane.width = 1424
    callback()

    assert window.main_pane.sash == 882
    assert window._initial_sash_applied is True
