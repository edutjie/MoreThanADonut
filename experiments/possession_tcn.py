"""Temporal convolution challengers for team possession."""

import torch
from torch import nn
from torch.nn import functional as F

from sequence_possession import SequenceModel


class DilatedConvolutions(nn.Module):
    def __init__(self, width, dilations):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in self.layers:
            x = x + self.dropout(F.gelu(layer(x)))
        return x


class PossessionTCN(SequenceModel):
    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        self.project = nn.Linear(2 * len(mean), 48)
        self.blocks = DilatedConvolutions(48, (1, 2, 4, 8, 16, 32))
        self.head = nn.Linear(48, 3)

    def forward(self, features, padding=None):
        x = self.project(self.standardized_inputs(features))
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)


class PlayerTCN(nn.Module):
    """Shared trajectory encoder with separate attention pools for the two tasks."""

    def __init__(self):
        super().__init__()
        width = 16
        self.project = nn.Linear(9, width)
        self.tracks = DilatedConvolutions(width, (1, 2, 4, 8, 5))
        self.attention = nn.ModuleList(nn.MultiheadAttention(width, 2, batch_first=True)
                                       for _ in range(2))
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(2 * width, 32), nn.LayerNorm(32),
                                                 nn.GELU(), nn.Linear(32, 1))
                                   for _ in range(2))

    def forward(self, objects, padding=None):
        batch, frames, objects_count, _ = objects.shape
        present = objects[..., 5] > 0
        present = F.max_pool1d(present.transpose(1, 2).float(), 9, stride=1, padding=4)
        hidden = present.transpose(1, 2).reshape(-1, objects_count) == 0
        hidden[:, -1] = False  # The ball token carries visibility even without coordinates.
        x = self.project(objects.nan_to_num())
        x = x.permute(0, 2, 3, 1).flatten(0, 1)
        x = self.tracks(x).view(batch, objects_count, -1, frames).permute(0, 3, 1, 2)
        tokens = x.reshape(batch * frames, objects_count, -1)
        ball = tokens[:, -1:]
        logits = []
        for attention, head in zip(self.attention, self.heads):
            pooled, _ = attention(ball, tokens, tokens, key_padding_mask=hidden, need_weights=False)
            logits.append(head(torch.cat((ball, pooled), dim=-1)).reshape(batch, frames))
        dead, home = logits
        live = F.logsigmoid(-dead)
        return torch.stack((live + F.logsigmoid(home), live + F.logsigmoid(-home),
                            F.logsigmoid(dead)), dim=-1)
