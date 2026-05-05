"""
ResNet9 Model Architecture for Plant Disease Classification.

This module defines a lightweight ResNet-style CNN used to classify
38 plant disease categories from leaf images.
"""

import torch.nn as nn


def ConvBlock(in_channels, out_channels, pool=False):
    """
    Creates a convolutional block with BatchNorm and ReLU activation.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool         (bool): If True, appends a MaxPool2d(4) layer.

    Returns:
        nn.Sequential: The convolutional block.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """
    A lightweight 9-layer residual network for image classification.

    Architecture:
        - 4 convolutional blocks (conv1–conv4)
        - 2 residual connections (res1, res2)
        - 1 classifier head (MaxPool → Flatten → Linear)

    Args:
        in_channels   (int): Number of image channels (3 for RGB).
        num_diseases  (int): Number of output disease classes.
    """

    def __init__(self, in_channels, num_diseases):
        super().__init__()

        # Encoder blocks
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)    # → 128 x 64 x 64
        self.res1  = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

        self.conv3 = ConvBlock(128, 256, pool=True)   # → 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)   # → 512 x 4  x 4
        self.res2  = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, xb):
        """
        Forward pass through the network.

        Args:
            xb: Input batch tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_diseases).
        """
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out    # Residual connection 1
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out    # Residual connection 2
        out = self.classifier(out)
        return out
