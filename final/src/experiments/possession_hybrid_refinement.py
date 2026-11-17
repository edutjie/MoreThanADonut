'External calibration for the baseline-stoppage spatial hybrid.'


import ctypes


import subprocess


from pathlib import Path


import numpy as np


COSTS = (.5, 1., 2.5, 5., 10.)


STRENGTHS = (0., 1., 2.)


class LiveDecoder:
    def __init__(self, root):
        source = Path(__file__).with_name("possession_hybrid_dp.c")
        binary = root.resolve() / "hybrid-live.so"
        if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(source), "-o", str(binary)], check=True)
        self.lib = ctypes.CDLL(str(binary))
        floating = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        dead = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        labels = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        self.lib.hybrid_live.argtypes = [floating, floating, dead, ctypes.c_int, labels]
        self.lib.hybrid_live.restype = ctypes.c_int

    def __call__(self, p, dead, fps, cost, strength=0, physics=None):
        if p.shape != (len(dead), 2) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("expected finite nonnegative H/A probabilities on the mask grid")
        if fps <= 0 or cost < 0 or strength < 0 or not np.isfinite([fps, cost, strength]).all():
            raise ValueError("expected positive frame rate and finite nonnegative transition parameters")
        if not len(dead):
            return np.empty(0, dtype=np.int8)
        evidence = np.ascontiguousarray(np.log(np.maximum(p, 1e-6)) * (10 / fps), dtype=np.float64)
        adjustment = np.zeros_like(p) if physics is None else physics
        if adjustment.shape != p.shape or not np.isfinite(adjustment).all():
            raise ValueError("physics costs must be finite and match the H/A frame grid")
        costs = np.ascontiguousarray(np.maximum(cost + strength * adjustment, 0), dtype=np.float64)
        mask = np.ascontiguousarray(dead, dtype=np.uint8)
        result = np.empty(len(dead), dtype=np.int8)
        if self.lib.hybrid_live(evidence, costs, mask, len(dead), result):
            raise MemoryError("could not allocate live decoder")
        return result


def restore(values, sample, length):
    return np.column_stack([np.interp(np.arange(length), sample, side) for side in values.T])


def team_p(p):
    return p[:, :2] / np.maximum(p[:, :2].sum(axis=1, keepdims=True), 1e-12)
