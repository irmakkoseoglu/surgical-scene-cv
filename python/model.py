"""U-Net with an ImageNet-pretrained ResNet encoder.

Kept deliberately simple (Conv, BatchNorm, ReLU, bilinear Upsample, Concat) so that the
exported ONNX graph runs in OpenCV's DNN module from C++ without custom layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_block(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, num_classes: int, encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        backbone = getattr(torchvision.models, encoder)(weights=weights)
        ch = [64, 64, 128, 256, 512]  # resnet18 / resnet34
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2
        self.pool = backbone.maxpool                                              # 1/4
        self.enc1, self.enc2, self.enc3, self.enc4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.dec4 = UpBlock(ch[4], ch[3], 256)  # 1/16
        self.dec3 = UpBlock(256, ch[2], 128)    # 1/8
        self.dec2 = UpBlock(128, ch[1], 64)     # 1/4
        self.dec1 = UpBlock(64, ch[0], 32)      # 1/2
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(32, 16),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        d = self.dec4(x4, x3)
        d = self.dec3(d, x2)
        d = self.dec2(d, x1)
        d = self.dec1(d, x0)
        return self.final(d)


class DeployWrapper(nn.Module):
    """Takes RGB in [0, 1] and applies ImageNet normalisation inside the graph,
    so the C++ side only needs cv::dnn::blobFromImage(img, 1/255, size, 0, swapRB=true)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)
