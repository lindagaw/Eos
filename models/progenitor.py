"""Progenitor model for ADDA."""

import torch.nn.functional as F
from torch import nn


class Progenitor(nn.Module):
    """Progenitor progenitorEncoder model for ADDA."""

    def __init__(self):
        """Init Progenitor progenitorEncoder."""
        super(Progenitor, self).__init__()

        self.restored = False


        # 1st conv layer
        # input [1 x 28 x 28]
        # output [20 x 12 x 12]
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 2nd conv layer
        # input [20 x 12 x 12]
        # output [50 x 4 x 4]
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.dropout2 = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, input):
        """Forward the Progenitor."""
        conv_out = F.relu(self.pool1(self.conv1(input)))
        conv_out = F.relu(self.pool2(self.dropout2(self.conv2(conv_out))))

        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
