"""Standard 3D U-Net baseline (Cicek et al. 2016 style) for the Phase 4 comparison
table. Deliberately vanilla -- no attention, no transformer blocks -- so it serves as
a clean CNN reference point against SegFormer3D and SegFormer3D+boundary.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 3, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for c in chs[:-1]:
            self.encoders.append(ConvBlock3D(in_ch, c))
            self.pools.append(nn.MaxPool3d(2))
            in_ch = c

        self.bottleneck = ConvBlock3D(chs[-2], chs[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose3d(chs[i + 1], chs[i], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(chs[i] * 2, chs[i]))

        self.classifier = nn.Conv3d(chs[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet3D(in_channels=4, num_classes=3, base_channels=32, depth=4)
    print(f"UNet3D params: {model.num_parameters() / 1e6:.2f}M")
    dummy = torch.randn(1, 4, 128, 128, 128)
    print(f"output: {tuple(model(dummy).shape)}")
