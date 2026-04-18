"""Fast Style Transfer CNN per Johnson et al. (2016).

Encoder (3 downsampling convs) -> 9 residual blocks -> Decoder (2 upsampling
convs + output conv). Uses Instance Normalization with affine parameters and
reflection padding on every conv. Output activation is tanh so the produced
tensor is in [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """Reflection-padded conv layer.

    The conv itself uses no padding -- padding is handled by the preceding
    ``nn.ReflectionPad2d(kernel // 2)`` so edge artifacts are avoided.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class UpsampleConvLayer(nn.Module):
    """Nearest-neighbor upsample followed by a reflection-padded conv.

    This avoids the checkerboard artifacts typical of transposed convs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        upsample: int,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample and self.upsample != 1:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    """Two-conv residual block with Instance Normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class TransformNet(nn.Module):
    """Johnson et al. Fast Style Transfer network.

    - Encoder: 9x9 conv (stride 1) -> 3x3 conv (stride 2) -> 3x3 conv (stride 2)
    - Residual core: 9 residual blocks at 128 channels
    - Decoder: nearest-upsample + 3x3 conv (x2) -> 9x9 conv (stride 1)
    - All normalization is Instance Normalization (never Batch).
    - Output activation: tanh -> range [-1, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual blocks (9)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)

        # Decoder
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return torch.tanh(y)
