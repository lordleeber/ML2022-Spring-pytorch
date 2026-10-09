# Student networks for docs/tools/hw13_exp.py (--student NAME). Every entry must stay <= 100,000 parameters.
#   sample  HW13.ipynb cell [19] (model.py)
#   dw      MobileNet-v1 style: a strided 3x3 stem, then depthwise-separable blocks (model.py's dwpw_conv + BN/ReLU)
#   mbv2    MobileNet-v2 style: inverted residual blocks (1x1 expand -> 3x3 depthwise -> 1x1 project, skip when shapes match)
#   plain   the same layout as dw (same strides, depth, BN/ReLU) with ordinary 3x3 convolutions, widths chosen
#           so the parameter count matches dw: the control for "is depthwise-separable better at equal size?"
import torch.nn as nn

from model import get_student_model, dwpw_conv


def conv_bn(cin, cout, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def dw_block(cin, cout, stride):
    # dwpw_conv from model.py, with BN + ReLU after each of its two convolutions
    dw, pw = dwpw_conv(cin, cout, 3, stride=stride, padding=1)
    return nn.Sequential(dw, nn.BatchNorm2d(cin), nn.ReLU(inplace=True), pw, nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Head(nn.Module):
    def __init__(self, body, cout):
        super().__init__()
        self.body = body
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cout, 11)

    def forward(self, x):
        return self.fc(self.pool(self.body(x)).flatten(1))


# (out_channels, stride) after the stem; the stem is 3 -> STEM, stride 2 (224 -> 112)
DW_STEM = 24
DW_CFG = [(32, 1), (64, 2), (64, 1), (96, 2), (96, 1), (128, 2), (128, 1), (128, 1), (144, 2)]


def dw_net():
    layers, c = [conv_bn(3, DW_STEM, 3, 2)], DW_STEM
    for cout, s in DW_CFG:
        layers.append(dw_block(c, cout, s))
        c = cout
    return Head(nn.Sequential(*layers), c)


PLAIN_STEM = 16
PLAIN_CFG = [(16, 1), (24, 2), (24, 1), (32, 2), (32, 1), (40, 2), (44, 1), (48, 1), (56, 2)]


def plain_net():
    layers, c = [conv_bn(3, PLAIN_STEM, 3, 2)], PLAIN_STEM
    for cout, s in PLAIN_CFG:
        layers.append(conv_bn(c, cout, 3, s))
        c = cout
    return Head(nn.Sequential(*layers), c)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        mid = cin * expand
        self.use_skip = stride == 1 and cin == cout
        self.block = nn.Sequential(
            conv_bn(cin, mid, 1),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout),   # linear bottleneck: no ReLU
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_skip else out


MB_STEM = 16
# (expand, out_channels, repeats, first stride)
MB_CFG = [(1, 16, 1, 1), (4, 24, 2, 2), (4, 32, 2, 2), (4, 48, 2, 2), (4, 64, 1, 2)]
MB_LAST = 160


def mbv2_net():
    layers, c = [conv_bn(3, MB_STEM, 3, 2)], MB_STEM
    for t, cout, n, s in MB_CFG:
        for i in range(n):
            layers.append(InvertedResidual(c, cout, s if i == 0 else 1, t))
            c = cout
    layers.append(conv_bn(c, MB_LAST, 1))
    return Head(nn.Sequential(*layers), MB_LAST)


STUDENTS = {
    'sample': get_student_model,   # HW13.ipynb cell [19], 87,907 parameters
    'dw': dw_net,
    'plain': plain_net,
    'mbv2': mbv2_net,
}

if __name__ == '__main__':
    import torch
    for name, fn in STUDENTS.items():
        m = fn()
        n = sum(p.numel() for p in m.parameters())
        print(name, n, tuple(m(torch.zeros(2, 3, 224, 224)).shape))
