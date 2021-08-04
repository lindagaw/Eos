"""Progenitor model for ADDA."""

import torch.nn.functional as F
from torch import nn


class Progenitor(nn.Module):
    """Progenitor progenitorEncoder model for ADDA."""

    def __init__(self):
        """Init Progenitor progenitorEncoder."""
        super(Progenitor, self).__init__()

        self.restored = False

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool6 = nn.MaxPool2d(kernel_size=1)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3)
        self.pool9 = nn.MaxPool2d(kernel_size=1)

        self.conv10 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3)
        self.pool12 = nn.MaxPool2d(kernel_size=1)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3)
        self.pool15 = nn.MaxPool2d(kernel_size=1)



        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 65)

    def forward(self, input):
        """Forward the Progenitor."""
        conv_out = F.relu(self.pool3(self.conv2((self.conv1(input)))))
        conv_out = F.relu(self.pool6(self.conv5((self.conv4(conv_out)))))
        conv_out = F.relu(self.pool9(self.conv8((self.conv7(conv_out)))))
        conv_out = F.relu(self.pool12(self.conv11((self.conv10(conv_out)))))
        conv_out = F.relu(self.pool15(self.conv14((self.conv13(conv_out)))))

        out = self.fc3(self.fc2(self.fc1(conv_out)))
        return out
