"""Synthetic grid, same-team restart and exhaustive small HSMM checks."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as m
from experiments.possession_execution import fixed
from experiments.possession_execution_verify import synthetic_grid
from experiments.possession_round2_decoder import Decoder
from experiments.possession_round2_segments import snap


def objective(labels, evidence, prior, tail):
    total = evidence[np.arange(len(labels)), labels].sum()
    segments = m.segments(np.asarray(labels), [(0, len(labels))])
    for i, (a, b, team) in enumerate(segments):
        d = b - a
        total += prior[min(d, len(prior)) - 1, team] + max(0, d - len(prior)) * tail[team]
        if i:
            total -= 1.3 if m.D in (team, segments[i - 1][2]) else 0.8
    return total


def main(output):
    synthetic_grid(output)
    decoder = Decoder(output)
    rng = np.random.default_rng(203)
    for fps in (10, 25):
        for n in (1, 20, 251):
            p = rng.dirichlet([1, 1, 1], size=n)
            for values in (p, p.astype(np.float32), np.full_like(p, 1/3)):
                assert np.array_equal(decoder(values, fps), fixed(values, fps))
    possibilities = list(product(range(3), repeat=8))
    for _ in range(8):
        p = rng.dirichlet([1, 1, 1], size=8)
        prior = rng.normal(size=(3, 3))
        tail = -rng.random(3)
        got = decoder(p, 10, (0.8, 1.3), (prior, tail))
        e = np.log(p)
        assert np.isclose(objective(got, e, prior, tail),
                          max(objective(choice, e, prior, tail) for choice in possibilities))
    labels = np.array([m.H] * 30 + [m.D] * 30 + [m.H] * 30)
    p = np.eye(3)[labels] * 0.98 + 0.01
    neutral = (np.zeros((8, 3)), np.zeros(3))
    for pred in (decoder(p, 25), decoder(p, 25, duration=neutral),
                 snap(decoder(p, 25), np.zeros(len(p)), 15)):
        assert np.array_equal(pred, labels)
        separated = np.r_[pred, m.X, pred]
        assert len(m.possession_spells(separated, m.valid_runs(separated))) == 4
        assert len(m.transitions(separated, m.valid_runs(separated))) == 4
    local_touch = np.zeros(len(p))
    local_touch[57] = 1
    shifted = snap(labels, local_touch, 5)
    assert m.transitions(shifted, [(0, len(p))]) == [(30, m.H, m.D), (57, m.D, m.H)]
    print("Original and native-grid Viterbi agree; HSMM matches exhaustive enumeration; "
          "stoppage, halftime and CSV/grid fixtures pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
