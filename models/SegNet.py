import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class ConvBnRelu2d(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SegNet(nn.Module):
    def __init__(self, n_class, in_channels=6):
        super(SegNet, self).__init__()
        
        chs = [32,64,64,128,128]

        self.down1 = nn.Sequential(
            ConvBnRelu2d(in_channels, chs[0]),
            ConvBnRelu2d(chs[0], chs[0]),
        )
        self.down2 = nn.Sequential(
            ConvBnRelu2d(chs[0], chs[1]),
            ConvBnRelu2d(chs[1], chs[1]),
        )
        self.down3 = nn.Sequential(
            ConvBnRelu2d(chs[1], chs[2]),
            ConvBnRelu2d(chs[2], chs[2]),
            ConvBnRelu2d(chs[2], chs[2])
        )
        self.down4 = nn.Sequential(
            ConvBnRelu2d(chs[2], chs[3]),
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[3])
        )
        self.down5 = nn.Sequential(
            ConvBnRelu2d(chs[3], chs[4]),
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[4])
        )
        self.up5 = nn.Sequential(
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[3])
        )
        self.up4 = nn.Sequential(
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[2])
        )
        self.up3 = nn.Sequential(
            ConvBnRelu2d(chs[2], chs[2]),
            ConvBnRelu2d(chs[2], chs[2]),
            ConvBnRelu2d(chs[2], chs[1])
        )
        self.up2 = nn.Sequential(
            ConvBnRelu2d(chs[1], chs[1]),
            ConvBnRelu2d(chs[1], chs[0])
        )
        self.up1 = nn.Sequential(
            ConvBnRelu2d(chs[0], chs[0]),
            ConvBnRelu2d(chs[0], n_class)
        )

    def forward(self, rgb,thr):
        x = torch.cat([rgb,thr],dim=1)
        
        x       = self.down1(x)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down2(x)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down3(x)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down4(x)
        x, ind4 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down5(x)
        x, ind5 = F.max_pool2d(x, 2, 2, return_indices=True)

        x       = F.max_unpool2d(x, ind5, 2, 2)
        x       = self.up5(x)
        x       = F.max_unpool2d(x, ind4, 2, 2)
        x       = self.up4(x)
        x       = F.max_unpool2d(x, ind3, 2, 2)
        x       = self.up3(x)
        x       = F.max_unpool2d(x, ind2, 2, 2)
        x       = self.up2(x)
        x       = F.max_unpool2d(x, ind1, 2, 2)
        x       = self.up1(x)

        return x
