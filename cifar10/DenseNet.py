import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, depth, growth_rate=12, reduction=0.5, num_classes=10,in_through=3):
        super().__init__()
        self.growth_rate = growth_rate
        # 计算每个dense块层数
        n_blocks = (depth - 4) // 6
        # 初始通道
        in_channels = 2 * growth_rate
        # CIFAR10 stem 3x3无大下采样
        self.stem = nn.Conv2d(in_through, in_channels, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense(block, in_channels, n_blocks)
        in_channels += n_blocks * growth_rate
        out_channels = int(in_channels * reduction)
        self.trans1 = Transition(in_channels, out_channels)
        in_channels = out_channels

        self.dense2 = self._make_dense(block, in_channels, n_blocks)
        in_channels += n_blocks * growth_rate
        out_channels = int(in_channels * reduction)
        self.trans2 = Transition(in_channels, out_channels)
        in_channels = out_channels

        self.dense3 = self._make_dense(block, in_channels, n_blocks)
        in_channels += n_blocks * growth_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.head = nn.Linear(in_channels, num_classes)

    def _make_dense(self, block, in_channels, n_blocks):
        layers = []
        for _ in range(n_blocks):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# 常用预定义
def densenet_bc_40(num_classes=10,in_through=3):
    # DenseNet-BC-40, k=12
    return DenseNet(Bottleneck, depth=40, growth_rate=12, reduction=0.5, num_classes=num_classes,in_through=in_through)

def densenet_bc_100(num_classes=10,in_through=3):
    # DenseNet-BC-100, k=12
    return DenseNet(Bottleneck, depth=100, growth_rate=12, reduction=0.5, num_classes=num_classes,in_through=in_through)