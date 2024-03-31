# 构建与nnUNet中unet相同的UNet，以便修改

import torch
import torch.nn as nn

class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_prob=0.5):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.all_modules = nn.Sequential(
            self.conv,
            self.norm,
            self.nonlin
        )

    def forward(self, x):
        x = self.all_modules(x)
        return x

class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackedConvBlocks, self).__init__()
        self.convs = nn.Sequential(
            ConvDropoutNormReLU(in_channels, out_channels),
            ConvDropoutNormReLU(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class PlainConvEncoder(nn.Module):
    def __init__(self):
        super(PlainConvEncoder, self).__init__()
        self.stages = nn.Sequential(
            nn.Sequential(
                StackedConvBlocks(1, 32)
            ),
            nn.Sequential(
                StackedConvBlocks(32, 64)
            ),
            nn.Sequential(
                StackedConvBlocks(64, 128)
            ),
            nn.Sequential(
                StackedConvBlocks(128, 256)
            ),
            nn.Sequential(
                StackedConvBlocks(256, 512)
            ),
            nn.Sequential(
                StackedConvBlocks(512, 512)
            ),
            nn.Sequential(
                StackedConvBlocks(512, 512)
            )
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return features

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.encoder = PlainConvEncoder()
        self.stages = nn.ModuleList([
            nn.Sequential(
                StackedConvBlocks(1024, 512),
                StackedConvBlocks(512, 512)
            ),
            StackedConvBlocks(512, 256),
            StackedConvBlocks(256, 128),
            StackedConvBlocks(128, 64)
        ])

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        for i, stage in enumerate(self.stages):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, features[i + 1]], dim=1)
            x = stage(x)
        return x

class PlainConvUNet(nn.Module):
    def __init__(self):
        super(PlainConvUNet, self).__init__()
        self.encoder = PlainConvEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

# 创建模型
model = PlainConvUNet()
# 打印模型结构
print(model)
