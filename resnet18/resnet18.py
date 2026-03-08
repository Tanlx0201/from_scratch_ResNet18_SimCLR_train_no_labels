import torch
import torch.nn as nn
from .basicblock import BasicBlock


class ResNet(nn.Module):
    """ResNet-18 (from scratch) with an option to return features.

    - If return_features=True: returns pooled features of shape [B, 512]
    - Else: returns logits of shape [B, num_classes]
    """

    def __init__(self, block, layers, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Keep the classifier head for supervised training
        self.fc = nn.Linear(512, num_classes) if num_classes is not None else nn.Identity()

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)

        if return_features:
            return feats

        return self.fc(feats)


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
