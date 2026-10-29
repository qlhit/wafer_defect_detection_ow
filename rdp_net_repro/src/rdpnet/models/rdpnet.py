from collections import OrderedDict

import torch
import torch.nn as nn


class SKAttention(nn.Module):
    def __init__(self, channel: int = 512, kernels=(1, 3, 5, 7), reduction: int = 16, group: int = 1, L: int = 32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                ("bn", nn.BatchNorm2d(channel)),
                ("relu", nn.ReLU()),
            ]))
            for k in kernels
        ])
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

        # spatial attention
        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = x.size()
        conv_outs = [conv(x) for conv in self.convs]  # list of (bs,c,h,w)
        feats = torch.stack(conv_outs, 0)  # (k,bs,c,h,w)

        # fuse
        U = sum(conv_outs)  # (bs,c,h,w)

        # reduction channel
        S = U.mean(-1).mean(-1)  # (bs,c)
        Z = self.fc(S)  # (bs,d)

        # calculate attention weight
        weights = [fc(Z).view(bs, c, 1, 1) for fc in self.fcs]  # list (bs,c,1,1)
        attention_weights = torch.stack(weights, 0)  # (k,bs,c,1,1)
        attention_weights = self.softmax(attention_weights)

        # fuse
        V = (attention_weights * feats).sum(0)  # (bs,c,h,w)

        # spatial attention
        US = self.conv1(V)
        US = self.conv2(US)
        US = self.activation(US)
        US = torch.unsqueeze(US, 2)  # (bs,4,1,h,w)
        US = torch.permute(US, [1, 0, 2, 3, 4])  # (4,bs,1,h,w)
        VS = (US * feats).sum(0)

        V = V + VS
        return V


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.skattention = SKAttention(channel=out_ch, reduction=16)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride == 2 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.skattention(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class RDPNet(nn.Module):
    def __init__(self, output_dim: int = 8, in_channels: int = 1, apply_sigmoid: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_1_1 = BasicBlock(in_ch=64, out_ch=64, stride=1)
        self.block_1_2 = BasicBlock(in_ch=64, out_ch=64, stride=1)

        self.block_2_1 = BasicBlock(in_ch=64, out_ch=128, stride=2)
        self.block_2_2 = BasicBlock(in_ch=128, out_ch=128, stride=1)

        self.block_3_1 = BasicBlock(in_ch=128, out_ch=256, stride=2)
        self.block_3_2 = BasicBlock(in_ch=256, out_ch=256, stride=1)

        self.block_4_1 = BasicBlock(in_ch=256, out_ch=512, stride=2)
        self.block_4_2 = BasicBlock(in_ch=512, out_ch=512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        self._apply_sigmoid = apply_sigmoid
        self._sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block_1_1(x)
        x = self.block_1_2(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)

        x = self.block_4_1(x)
        x = self.block_4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self._apply_sigmoid:
            x = self._sigmoid(x)  # multi-label outputs in [0,1]
        return x
