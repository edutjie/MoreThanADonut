"""Synthetic checks for the pre-kickoff dead-ball rule. Run: python -m unittest discover -s tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kickoff import (  # noqa: E402
    decode_after_kickoff, detect_kickoff, kickoff_offset, kickoff_windows, prekickoff_dead,
)

FPS = 25
CENTRE = (52.65, 34.0)


def resting_then_kicked(n=300, kick=150, speed=8.0, jitter=0.05, seed=0):
    """Ball at the centre spot with tracking jitter, struck at `kick` and moving at `speed` m/s."""
    rng = np.random.default_rng(seed)
    ball = np.tile(np.asarray(CENTRE), (n, 1)) + rng.normal(0, jitter, (n, 2))
    steps = np.arange(n - kick)[:, None] / FPS
    ball[kick:] = np.asarray(CENTRE) + steps * np.array([speed, 0.0]) + np.array([0.4, 0.0])
    return ball


class DetectKickoff(unittest.TestCase):
    def test_strike_from_rest(self):
        frame, method = detect_kickoff(resting_then_kicked(), (100, 220, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (150, "tracking"))

    def test_missing_ball_falls_back_to_tag(self):
        ball = np.full((300, 2), np.nan)
        self.assertEqual(detect_kickoff(ball, (100, 220, 160), CENTRE, FPS), (160, "tag"))

    def test_rest_away_from_spot_is_not_a_kickoff(self):
        ball = resting_then_kicked() + np.array([5.0, 0.0])
        self.assertEqual(detect_kickoff(ball, (100, 220, 160), CENTRE, FPS), (160, "tag"))

    def test_nudge_that_returns_is_ignored(self):
        ball = resting_then_kicked(kick=180)
        ball[120:126] += np.array([0.5, 0.0])  # placed or nudged, then back on the spot
        frame, method = detect_kickoff(ball, (100, 240, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (180, "tracking"))

    def test_small_drift_without_confirmation_is_ignored(self):
        ball = resting_then_kicked(kick=180)
        ball[120:] = np.asarray(CENTRE) + np.array([0.6, 0.0])  # moved 0.6 m and stays
        _, method = detect_kickoff(ball, (100, 240, 160), CENTRE, FPS)
        self.assertEqual(method, "tag")

    def test_gap_before_motion_starts_after_last_rest(self):
        ball = resting_then_kicked()
        ball[140:155] = np.nan  # ball lost from frame 140, seen moving from 155
        frame, method = detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (140, "tracking"))

    def test_slow_kicks_without_a_step_are_detected(self):
        for speed in (3.0, 5.0, 7.0, 10.0):
            ball = resting_then_kicked(speed=speed)
            ball[150:] -= np.array([0.4, 0.0])  # no jump on the strike frame
            frame, method = detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)
            self.assertEqual(method, "tracking", speed)
            # first motion is 150 -> 151; a frame either side is free in the metric
            self.assertLessEqual(frame, 152, speed)
            self.assertGreaterEqual(frame, 149, speed)

    def test_creeping_start_is_not_detected_late(self):
        ball = resting_then_kicked(kick=160)
        ball[151:160] = np.asarray(CENTRE) + np.arange(1, 10)[:, None] * np.array([0.1, 0.0])
        frame, method = detect_kickoff(ball, (100, 220, 200), CENTRE, FPS)
        self.assertEqual(method, "tracking")
        # first motion at 151; a 2.5 m/s creep inside the jitter band costs at most two frames
        self.assertLessEqual(frame, 153)

    def test_return_after_window_end_still_rejects_a_nudge(self):
        ball = np.tile(np.asarray(CENTRE), (300, 1))
        ball[205:215] += np.linspace(0.5, 2.0, 10)[:, None] * np.array([1.0, 0.0])
        # back on the spot from frame 215, after the window stop at 212
        frame, method = detect_kickoff(ball, (100, 212, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (160, "tag"))

    def test_frozen_ball_that_reappears_moving_from_the_spot(self):
        ball = resting_then_kicked(kick=150, speed=12.0)
        ball[:150] = np.array([100.0, 30.0])  # tracker holds the last live position, far away
        ball[150] = np.asarray(CENTRE) + np.array([0.3, 0.0])
        frame, method = detect_kickoff(ball, (100, 220, 200), CENTRE, FPS)
        self.assertEqual(method, "tracking")
        self.assertIn(frame, (150, 151))

    def test_hidden_ball_that_appears_moving_from_the_spot(self):
        ball = resting_then_kicked(kick=150, speed=8.0)
        ball[:150] = np.nan
        frame, method = detect_kickoff(ball, (100, 220, 200), CENTRE, FPS)
        self.assertEqual(method, "tracking")
        self.assertIn(frame, (150, 151))

    def test_radius_argument_overrides_default(self):
        ball = resting_then_kicked() + np.array([3.5, 0.0])
        self.assertEqual(detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)[1], "tag")
        self.assertEqual(detect_kickoff(ball, (100, 220, 160), CENTRE, FPS, radius=4.0)[1], "tracking")

    def test_noisy_rest_sighting_is_not_the_strike(self):
        for outlier in (140, 145, 148):
            ball = resting_then_kicked()
            ball[outlier, 1] += 0.2
            frame, method = detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)
            self.assertEqual(method, "tracking", outlier)
            self.assertGreaterEqual(frame, 149, outlier)

    def test_jittery_rest_is_rarely_early(self):
        early = 0
        for seed in range(50):
            ball = resting_then_kicked(jitter=0.08, seed=seed)
            frame, _ = detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)
            early += frame < 148
        self.assertLessEqual(early, 5)

    def test_slow_push_that_returns_after_reaching_is_rejected(self):
        ball = np.tile(np.asarray(CENTRE), (300, 1))
        ball[150:164] += (np.arange(1, 15) * 0.12)[:, None] * np.array([1.0, 0.0])
        ball[164:176] = np.asarray(CENTRE) + np.array([1.6, 0.0])
        frame, method = detect_kickoff(ball, (100, 220, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (160, "tag"))

    def test_checks_look_past_the_window_end(self):
        ball = resting_then_kicked(kick=210)
        ball[211:232] = np.nan  # long gap right after the strike
        ball[232:] = np.asarray(CENTRE)  # back on the spot after the gap: not a kick
        frame, method = detect_kickoff(ball, (100, 213, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (160, "tag"))

    def test_strike_outside_window_is_ignored(self):
        frame, method = detect_kickoff(resting_then_kicked(kick=250), (100, 220, 160), CENTRE, FPS)
        self.assertEqual((frame, method), (160, "tag"))


class PrekickoffDead(unittest.TestCase):
    def setUp(self):
        self.labels = np.array(list("XXXX" + "HHAA" + "XX" + "HHAAHH" + "XX"))
        # graded halves: 4..7 and 10..15
        self.blocks = [(1, 4, 7), (2, 10, 15)]

    def test_only_graded_frames_before_kickoff_change(self):
        out, changed = prekickoff_dead(self.labels, self.blocks, {1: 3, 2: 13})
        self.assertEqual("".join(out), "XXXX" + "HHAA" + "XX" + "DDDAHH" + "XX")
        self.assertEqual(changed, {1: 0, 2: 3})

    def test_kickoff_at_block_start_is_a_no_op(self):
        out, changed = prekickoff_dead(self.labels, self.blocks, {1: 4, 2: 10})
        self.assertEqual("".join(out), "".join(self.labels))
        self.assertEqual(changed, {1: 0, 2: 0})

    def test_never_writes_x_and_stops_at_half_end(self):
        out, _ = prekickoff_dead(self.labels, self.blocks, {2: 99})
        self.assertEqual("".join(out[10:16]), "DDDDDD")
        self.assertEqual("".join(out[16:]), "XX")
        self.assertNotIn("X", "".join(out[4:8]) + "".join(out[10:16]))

    def test_missing_half_is_left_unchanged(self):
        out, changed = prekickoff_dead(self.labels, self.blocks, {})
        self.assertEqual("".join(out), "".join(self.labels))
        self.assertEqual(changed, {1: 0, 2: 0})

    def test_input_is_not_mutated(self):
        before = self.labels.copy()
        prekickoff_dead(self.labels, self.blocks, {2: 13})
        np.testing.assert_array_equal(self.labels, before)


class DecodeAfterKickoff(unittest.TestCase):
    def test_offset_is_clamped_to_the_half(self):
        kickoffs = {1: (3, "tag"), 2: (40, "tag"), 3: (99, "tag")}
        self.assertEqual(kickoff_offset(kickoffs, 1, 10, 30), 0)
        self.assertEqual(kickoff_offset(kickoffs, 2, 10, 50), 30)
        self.assertEqual(kickoff_offset(kickoffs, 3, 10, 50), 41)
        self.assertEqual(kickoff_offset(kickoffs, 4, 10, 50), 0)

    def test_decoder_runs_from_the_offset_only(self):
        seen = []

        def decode(k):
            seen.append(k)
            return np.array(["H"] * (6 - k))

        out = decode_after_kickoff(decode, 6, 2)
        self.assertEqual("".join(out), "DDHHHH")
        self.assertEqual(seen, [2])

    def test_zero_offset_matches_plain_decoding(self):
        out = decode_after_kickoff(lambda k: np.array(list("HAD")[k:]), 3, 0)
        self.assertEqual("".join(out), "HAD")

    def test_whole_half_before_kickoff_skips_the_decoder(self):
        out = decode_after_kickoff(lambda k: 1 / 0, 4, 4)
        self.assertEqual("".join(out), "DDDD")

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            decode_after_kickoff(lambda k: np.array(["H"]), 4, 1)


class KickoffWindows(unittest.TestCase):
    @staticmethod
    def tag(code, kind, half, start, end):
        return {"code": code, "start_frame": start, "end_frame": end,
                "tags": {"Team": "N/A", "Type": kind, "Half": half}}

    def test_parses_case_insensitively_and_pads(self):
        tags = [self.tag("SET PIECES", "Kick Off (Start)", "2nd Half", 1000, 1125),
                self.tag("SET PIECES", "Kick Off (After Goal)", "2nd Half", 5000, 5125),
                self.tag("SET PIECES", "kick off (start)", "1st half", 10, 135)]
        windows = kickoff_windows(tags, FPS)
        self.assertEqual(windows, {2: (950, 1175, 1062), 1: (0, 185, 72)})

    def test_duplicate_half_raises(self):
        tags = [self.tag("SET PIECES", "Kick Off (Start)", "2nd Half", 1000, 1125)] * 2
        with self.assertRaises(ValueError):
            kickoff_windows(tags, FPS)


if __name__ == "__main__":
    unittest.main()
