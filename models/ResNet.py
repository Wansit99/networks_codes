import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
from utils.utils import print_model

writer = SummaryWriter("/root/tf-logs")


class Residual(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            use1x1conv=False,
            stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        if use1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Residual_3_layers(nn.Module):
    def __init__(self, input_channels, output_channels, use1x1conv=False, stride=1):
        super().__init__()
        mid_channels = output_channels // 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, output_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
        )

        if use1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.bottleneck(X)

        # 如果改变通道数
        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(Y)


def resnet_block_3_layers(
        input_channels,
        output_channels,
        num_residuals,
        first_block=False):
    blocks = []
    for i in range(num_residuals):
        if i == 0:
            blocks.append(
                Residual_3_layers(
                    input_channels,
                    output_channels,
                    use1x1conv=True,
                    stride=2))
        else:
            blocks.append(Residual_3_layers(output_channels, output_channels))
    return blocks

def resnet_block(
        input_channels,
        output_channels,
        num_residuals,
        first_block=False):
    blocks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blocks.append(
                Residual(
                    input_channels,
                    output_channels,
                    use1x1conv=True,
                    stride=2))
        else:
            blocks.append(Residual(output_channels, output_channels))
    return blocks


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        net = nn.Sequential(
            b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Flatten(), nn.Linear(
                512, num_classes))
        self.net = net

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000):
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block_3_layers(64, 256, blocks[0], first_block=True))
        b3 = nn.Sequential(*resnet_block_3_layers(256, 512, blocks[1]))
        b4 = nn.Sequential(*resnet_block_3_layers(512, 1024, blocks[2]))
        b5 = nn.Sequential(*resnet_block_3_layers(1024, 2048, blocks[3]))
        net = nn.Sequential(
            b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Flatten(), nn.Linear(
                2048, num_classes))
        self.net = net

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x

def ResNet18(num_classes=1000):
    return ResNet18(num_classes)

def ResNet50(num_classes=1000):
    return ResNet([3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes=1000):
    return ResNet([3, 4, 23, 3], num_classes=num_classes)

def ResNet152(num_classes=1000):
    return ResNet([3, 8, 36, 3], num_classes=num_classes)

if __name__=='__main__':
    model = ResNet50()
    # print(model)
    #
    print_model(model)
    # input = torch.randn(1, 3, 224, 224)
    # out = model(input)
    # print(out.shape)

