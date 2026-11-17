'Evaluate bidirectional sequence models on the frozen external possession splits.'


import numpy as np


import torch


from torch import nn


from external_possession import LAG_SECONDS, tracking_features


WINDOW = 512


CONTEXT = 80


DECODER_COSTS = (5, 10)


class SequenceModel(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def standardized_inputs(self, features):
        missing = ~torch.isfinite(features)
        x = ((features - self.mean) / self.scale).masked_fill(missing, 0).clamp(-8, 8)
        return torch.cat((x, missing.to(x.dtype)), dim=-1)


def visibility_features(ball, fps):
    visible = np.isfinite(ball).all(axis=1)
    frames = np.arange(len(ball))
    last = np.maximum.accumulate(np.where(visible, frames, -1))
    age = np.where(last >= 0, np.log1p((frames - last) / fps), np.nan)
    return np.column_stack((visible, age)).astype(np.float32)


def sequence_features(positions, weights, teams, pitch, fps, continuous_motion=False, *,
                      observed_ball_motion=False):
    base, mirrored = tracking_features(positions, weights, teams, pitch, fps, LAG_SECONDS,
                                      continuous_motion=continuous_motion, observed_ball_motion=observed_ball_motion)
    visibility = visibility_features(positions[:, -1], fps)
    return np.column_stack((base, visibility)), np.column_stack((mirrored, visibility))
