from __future__ import annotations

import unittest

from auction_moment_assistant.state_machine import (
    RoundWatchStateMachine,
    ScreenState,
    StateFeatures,
)


def features(*, banner_dark: float = 0.2) -> StateFeatures:
    return StateFeatures(
        top_white=0.4,
        center_white=0.3,
        event_button_blue=0.0,
        personal_event_button_blue=0.6,
        round_start_banner_dark=banner_dark,
        final_skip_button_dark=0.0,
        final_skip_button_white=0.5,
        keypad_dark=0.0,
        bid_button_blue=0.7,
        left_saturation=30.0,
        left_white=0.4,
    )


class RoundWatchStateMachineTests(unittest.TestCase):
    def test_waits_for_banner_then_scans_each_round_once(self) -> None:
        machine = RoundWatchStateMachine(fallback_active_frames=5)
        machine.observe(ScreenState.LOBBY, features())
        started = machine.observe(ScreenState.EVENT_SELECTION, features())
        self.assertIn("session_started", [action.kind for action in started])

        first = machine.observe(
            ScreenState.ROUND_ACTIVE, features(banner_dark=0.60)
        )
        self.assertNotIn("scan_round", [action.kind for action in first])
        ready = machine.observe(
            ScreenState.ROUND_ACTIVE, features(banner_dark=0.20)
        )
        self.assertEqual(
            [(action.kind, action.round_number) for action in ready],
            [("scan_round", 1)],
        )
        self.assertFalse(
            any(
                action.kind == "scan_round"
                for action in machine.observe(
                    ScreenState.ROUND_ACTIVE, features(banner_dark=0.20)
                )
            )
        )
        machine.mark_scan_result(1, True)

        machine.observe(ScreenState.ROUND_RANKING, features())
        changed = machine.observe(ScreenState.EVENT_SELECTION, features())
        self.assertIn(
            ("round_changed", 2),
            [(action.kind, action.round_number) for action in changed],
        )
        actions = ()
        for _ in range(5):
            actions = machine.observe(
                ScreenState.ROUND_ACTIVE, features(banner_dark=0.20)
            )
        self.assertIn(
            ("scan_round", 2),
            [(action.kind, action.round_number) for action in actions],
        )

    def test_final_result_ends_session_once(self) -> None:
        machine = RoundWatchStateMachine()
        machine.observe(ScreenState.EVENT_SELECTION, features())
        ended = machine.observe(ScreenState.FINAL_RESULT, features())
        self.assertIn("session_ended", [action.kind for action in ended])
        repeated = machine.observe(ScreenState.FINAL_RESULT, features())
        self.assertNotIn("session_ended", [action.kind for action in repeated])

    def test_return_to_lobby_also_clears_a_missed_final(self) -> None:
        machine = RoundWatchStateMachine()
        machine.observe(ScreenState.EVENT_SELECTION, features())
        ended = machine.observe(ScreenState.LOBBY, features())
        self.assertIn("session_ended", [action.kind for action in ended])


if __name__ == "__main__":
    unittest.main()
