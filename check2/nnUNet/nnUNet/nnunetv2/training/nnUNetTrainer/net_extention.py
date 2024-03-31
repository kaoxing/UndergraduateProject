# author: kaoing
# date: 2024-3-26
# 拓展UNet网络
import numpy as np
import torch
from torch import nn


class ConvNormLeakyReLU(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio+1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio+1, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionBlock(nn.Module):
    """
    构建模块用于在空间上和通道上融合两类特征图像
    :return:
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False)
        self.channel_layer1 = ChannelAttention(input_channels)
        self.channel_layer2 = ChannelAttention(input_channels)
        self.spatial_layer1 = SpatialAttention()
        self.spatial_layer2 = SpatialAttention()
        self.conv3 = nn.Conv2d(2*input_channels, input_channels, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(2*input_channels, input_channels, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.conv6 = ConvNormLeakyReLU(3*input_channels, output_channels, kernel_size=1)

    def forward(self, input1, input2):
        x1 = self.conv1(input1)
        x2 = self.conv2(input2)

        channel_attention1 = self.channel_layer1(x1)
        channel_attention2 = self.channel_layer2(x2)
        channel_attention3 = torch.cat([channel_attention1, channel_attention2], dim=1)
        channel_attention3 = self.conv3(channel_attention3)

        mid1 = torch.cat([x1, x2], dim=1)
        mid1 = self.conv4(mid1)
        mid2 = mid1 * channel_attention3

        spatial_attention1 = self.spatial_layer1(x1)
        spatial_attention2 = self.spatial_layer2(x2)
        spatial_attention3 = torch.cat([spatial_attention1, spatial_attention2], dim=1)
        spatial_attention3 = self.conv5(spatial_attention3)
        mid3 = mid2 * spatial_attention3

        mid4 = torch.cat([x1, x2, mid3], dim=1)
        mid4 = self.conv6(mid4)

        return mid4


if __name__ == '__main__':
    data1 = torch.rand((1, 3, 3, 3))
    data2 = torch.rand((1, 3, 3, 3))

    fusion_block = FusionBlock(3, 3)
    output = fusion_block(data1, data2)
    print(output.shape)
    print(output)

    # data1 = torch.Tensor(np.array([[[[1, 3, 5], [7, 9, 11], [13, 15, 17]]]]))
    # data2 = torch.Tensor(np.array([[[[2, 4, 6], [8, 10, 12], [14, 16, 18]]]]))
    # data3 = torch.Tensor(np.array([[[[1], [2], [3]]]]))
    # data4 = data2*data3
    # print(data4)