"""Native-grid Viterbi and explicit-duration possession decoding."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np


class Decoder:
    def __init__(self, root):
        binary = root.resolve() / "round2-dp.so"
        source = Path(__file__).with_name("possession_round2_dp.c")
        if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(source), "-o", str(binary)],
                           check=True)
        lib = ctypes.CDLL(str(binary))
        floating = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        output = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        common = [floating, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        lib.viterbi.argtypes = [*common, output]
        lib.viterbi.restype = None
        lib.hsmm.argtypes = [*common, floating, floating, ctypes.c_int, output]
        lib.hsmm.restype = ctypes.c_int
        self.lib = lib

    def __call__(self, p, fps, costs=(5, 10), duration=None):
        # Preserve the original dtype at the log and scaling steps.
        evidence = np.ascontiguousarray(np.log(np.maximum(p, 1e-6)) * (10 / fps), dtype=float)
        if duration is None:
            out = np.empty(4 * len(p), dtype=np.int8)
            self.lib.viterbi(evidence, len(p), *costs, out)
            return out[:len(p)].copy()
        prior, slope = duration
        out = np.empty(len(p), dtype=np.int8)
        if self.lib.hsmm(evidence, len(p), *costs, prior, slope, len(prior), out):
            raise MemoryError("could not allocate HSMM work arrays")
        return out
