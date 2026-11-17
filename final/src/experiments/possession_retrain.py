"""Frozen spatial TCN definition and symmetric inference."""


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


from experiments.possession_retrain_data import frame_features, mirrored


WINDOW = 768


HALO = 144


CENTRAL = WINDOW - 2 * HALO


class SpatialTCN(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.project = nn.Linear(2 * len(mean), 96)
        self.layers = nn.ModuleList(nn.Conv1d(96, 96, 3, padding=d, dilation=d)
                                    for d in (1, 2, 4, 8, 16, 32, 64))
        self.norms = nn.ModuleList(nn.LayerNorm(96) for _ in self.layers)
        self.out = nn.Linear(96, 3)
        self.boundary = nn.Linear(96, 1)

    def forward(self, x):
        valid = torch.isfinite(x)
        z = torch.where(valid, ((x - self.mean) / self.scale).clamp(-10, 10), 0)
        hidden = F.gelu(self.project(torch.cat((z, valid.float()), dim=-1)))
        for layer, norm in zip(self.layers, self.norms):
            delta = layer(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = norm(hidden + F.dropout(F.gelu(delta), .1, self.training))
        return self.out(hidden), self.boundary(hidden).squeeze(-1)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state"]
    if saved.get("architecture", "tcn") != "tcn":
        raise ValueError("the final package requires its frozen spatial TCN")
    model = SpatialTCN(state["mean"], state["scale"]).to(device)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def predict(model, tracks, teams, pitch, source):
    normal = frame_features(tracks, teams, pitch, source)
    flipped, other = mirrored(tracks, teams, pitch)
    mirror = frame_features(flipped, other, pitch, source)
    output = np.empty((len(tracks), 3), dtype=np.float32)
    for start in range(0, len(tracks), CENTRAL):
        stop = min(len(tracks), start + CENTRAL)
        lo, hi = max(0, start - HALO), min(len(tracks), stop + HALO)
        x = torch.from_numpy(np.stack((normal[lo:hi], mirror[lo:hi]))).to(model.mean.device)
        p = model(x)[0].softmax(-1).cpu().numpy()
        output[start:stop] = .5 * (p[0, start-lo:stop-lo] + p[1, start-lo:stop-lo][:, [1, 0, 2]])
    return output
