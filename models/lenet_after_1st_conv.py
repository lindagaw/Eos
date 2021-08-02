"""LeNet_Conv_1_ model for ADDA."""

import torch.nn.functional as F
from torch import nn


class LeNet_Conv_1_Encoder(nn.Module):
    """LeNet_Conv_1_ encoder model for ADDA."""

    def __init__(self):
        """Init LeNet_Conv_1_ encoder."""
        super(LeNet_Conv_1_Encoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            # nn.Conv2d(1, 20, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2),
            # nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(50, 50, kernel_size=1),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(50, 50, kernel_size=1),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet_Conv_1_."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNet_Conv_1_Classifier(nn.Module):
    """LeNet_Conv_1_ classifier model for ADDA."""

    def __init__(self):
        """Init LeNet_Conv_1_ encoder."""
        super(LeNet_Conv_1_Classifier, self).__init__()
        self.fc2 = nn.Linear(500, 65)

    def forward(self, feat):
        """Forward the LeNet_Conv_1_ classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
