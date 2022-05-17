#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from .resnet import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BongNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, c_heads=8, r_heads=168, v_heads=11,
                 pretrained=False, resnet_path=''):
        super().__init__()
        self.dropout = dropout
        self.c_heads = c_heads
        self.r_heads = r_heads
        self.v_heads =v_heads
        self.resnet_path = resnet_path

        if layers == 50:
            resnet = resnet50(pretrained=pretrained, pth=self.resnet_path)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained, pth=self.resnet_path)
        elif layers == 152:
            resnet = resnet152(pretrained=pretrained, pth=self.resnet_path)
        else:
            resnet = resnet50(pretrained=pretrained, pth=self.resnet_path)

        if resnet.deep_base:
            resnet.conv1 = conv3x3(1, 64, stride=2)
        else:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        return_layers = {"layer4": "out"}
        self.backbone = create_feature_extractor(resnet, return_layers)

        self.avg_pool_c = nn.AvgPool2d(7, stride=1)
        self.avg_pool_r = nn.AvgPool2d(7, stride=1)
        self.avg_pool_v = nn.AvgPool2d(7, stride=1)

        self.fc_c = nn.Linear(2048, self.c_heads)
        self.fc_r = nn.Linear(2048, self.r_heads)
        self.fc_v = nn.Linear(2048, self.v_heads)

    def forward(self, x):
        features = self.backbone(x)
        x = features["out"]

        x_c = self.avg_pool_c(x)
        x_c = x_c.view(x_c.size(0), -1)
        x_c = self.fc_c(x_c)

        x_r = self.avg_pool_r(x)
        x_r = x_r.view(x_r.size(0), -1)
        x_r = self.fc_r(x_r)

        x_v = self.avg_pool_v(x)
        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.fc_v(x_v)

        return F.log_softmax(x_c), F.log_softmax(x_r), F.log_softmax(x_v)


if __name__ == '__main__':
    model = BongNet()
    print(model)

