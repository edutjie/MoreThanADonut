'Temporal convolution challengers for team possession.'


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
