'Compare models on the same repaired tracking, splits, and spell decoder.'


import numpy as np


import torch


from external_possession import decode_costs


from sequence_possession import CONTEXT, DECODER_COSTS, WINDOW


MIRROR = np.array([1, 0, 2, 3])


@torch.inference_mode()
def dense_scores(model, normal, mirror, device):
    central = WINDOW - 2 * CONTEXT
    p = np.empty((len(normal), 3), dtype=np.float32)
    for start in range(0, len(p), central):
        stop = min(len(p), start + central)
        left, right = max(0, start - CONTEXT), min(len(p), stop + CONTEXT)
        x = torch.from_numpy(np.stack((normal[left:right], mirror[left:right])))
        probs = model(x.to(device, dtype=torch.float32)).softmax(-1).cpu().numpy()
        merged = (probs[0] + probs[1][:, MIRROR[:3]]) / 2
        p[start:stop] = merged[start - left:stop - left]
    return p


def decode_sampled(p, frames, length, fps, start=0, turnover_adjustment=None):
    """Interpolate sampled probabilities to every frame and decode frames start..length-1."""
    restored = np.column_stack([np.interp(np.arange(length), frames, p[:, k]) for k in range(3)])
    costs = None if turnover_adjustment is None else turnover_adjustment[start:]
    return decode_costs(np.log(np.maximum(restored[start:], 1e-6)), fps, *DECODER_COSTS,
                        turnover_adjustment=costs)


def columns(size):
    return [f"f{i:03d}" for i in range(size)]
