"""U-Net with an ImageNet ResNet-34 encoder (decoder channels 256-128-64-32-16, as in the common U-Net/ResNet-34 design)."""

import torch, torch.nn as nn, torch.nn.functional as F, torchvision.models as tvm


def _blk(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class ResUNet34(nn.Module):
    def __init__(self, out_ch=2, imagenet_weights=None):
        super().__init__()
        r = tvm.resnet34(weights=None)
        if imagenet_weights:
            r.load_state_dict(torch.load(imagenet_weights, map_location="cpu"))
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # 1/2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = (
            r.layer1,
            r.layer2,
            r.layer3,
            r.layer4,
        )  # 1/4 64, 1/8 128, 1/16 256, 1/32 512
        self.d4, self.d3, self.d2, self.d1 = (
            _blk(512 + 256, 256),
            _blk(256 + 128, 128),
            _blk(128 + 64, 64),
            _blk(64 + 64, 32),
        )
        self.d0 = _blk(32, 16)
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        x0 = self.stem(x)
        x1 = self.l1(self.pool(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        y = self.d4(torch.cat([up(x4), x3], 1))
        y = self.d3(torch.cat([up(y), x2], 1))
        y = self.d2(torch.cat([up(y), x1], 1))
        y = self.d1(torch.cat([up(y), x0], 1))
        return self.head(self.d0(up(y)))
