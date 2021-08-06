import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import math



class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)
        


class ResConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),stride=1,padding=1
                                 ),
            torch.nn.BatchNorm2d(out_channels),
            Swish(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),stride=1,padding=1
                                 ),
            torch.nn.BatchNorm2d(out_channels),
            Swish(),
        ])
        self.shortcut = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(1,1),stride=1),
            torch.nn.BatchNorm2d(out_channels)
        ])
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = self.layer(x)
        return x + shortcut

    
class FusionBlock(torch.nn.Module):
    def __init__(self,channel,height,width):
        super(FusionBlock, self).__init__()
        self.weightT = torch.nn.Parameter(torch.rand(size=(channel,height,width),requires_grad = True))
        self.weightR = torch.nn.Parameter(torch.rand(size=(channel,height,width),requires_grad = True))
        
    def forward(self,thr,rgb):
        thrAlpha = torch.sigmoid(self.weightT)
        thrBeta = 1-thrAlpha
        
        rgbAlpha = torch.sigmoid(self.weightR)
        rgbBeta = 1-rgbAlpha
        
        thr_ = thrAlpha*thr + thrBeta*rgb
        rgb_ = rgbAlpha*rgb + rgbBeta*thr
        
        return thr_,rgb_

class MBConvIntertwinedUNet(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_features = [64,256,256,256,256]
        self.downsample = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

        self.rgbStem = MBConvBlock(in_planes=3,out_planes=self.n_features[0], expand_ratio=1,kernel_size=3,stride=1)
        self.thrStem = MBConvBlock(in_planes=3,out_planes=self.n_features[0], expand_ratio=1,kernel_size=3,stride=1)
        

        self.rgbLayer1 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[0],out_planes=self.n_features[1], expand_ratio=6,kernel_size=3,stride=1),
            MBConvBlock(in_planes=self.n_features[1],out_planes=self.n_features[1], expand_ratio=6,kernel_size=3,stride=1),
            ])
        
        self.thrLayer1 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[0],out_planes=self.n_features[1], expand_ratio=6,kernel_size=3,stride=1),
            MBConvBlock(in_planes=self.n_features[1],out_planes=self.n_features[1], expand_ratio=6,kernel_size=3,stride=1),
            ])

        self.rgbLayer2 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[1],out_planes=self.n_features[2], expand_ratio=6,kernel_size=3,stride=1),
            MBConvBlock(in_planes=self.n_features[2],out_planes=self.n_features[2], expand_ratio=6,kernel_size=3,stride=1),
            ])
        self.thrLayer2 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[1],out_planes=self.n_features[2], expand_ratio=6,kernel_size=3,stride=1),
            MBConvBlock(in_planes=self.n_features[2],out_planes=self.n_features[2], expand_ratio=6,kernel_size=3,stride=1),
            ])

        self.rgbLayer3 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[2],out_planes=self.n_features[3], expand_ratio=6,kernel_size=5,stride=1),
            MBConvBlock(in_planes=self.n_features[3],out_planes=self.n_features[3], expand_ratio=6,kernel_size=5,stride=1),
            ])
        self.thrLayer3 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[2],out_planes=self.n_features[3], expand_ratio=6,kernel_size=5,stride=1),
            MBConvBlock(in_planes=self.n_features[3],out_planes=self.n_features[3], expand_ratio=6,kernel_size=5,stride=1),
            ])
        
        self.rgbLayer4 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[3],out_planes=self.n_features[4], expand_ratio=6,kernel_size=5,stride=1),
            MBConvBlock(in_planes=self.n_features[4],out_planes=self.n_features[4], expand_ratio=6,kernel_size=5,stride=1),
            ])
        self.thrLayer4 = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[3],out_planes=self.n_features[4], expand_ratio=6,kernel_size=5,stride=1),
            MBConvBlock(in_planes=self.n_features[4],out_planes=self.n_features[4], expand_ratio=6,kernel_size=5,stride=1),
            ])

        self.bridge = torch.nn.Sequential(*[
            MBConvBlock(in_planes=self.n_features[4]*2,out_planes=self.n_features[4], expand_ratio=6,kernel_size=5,stride=1)

            ])
        
#         self.fuse1 = FusionBlock(channel=self.n_features[0],height=128//2,width=256//2)
#         self.fuse2 = FusionBlock(channel=self.n_features[1],height=128//4,width=256//4)
#         self.fuse3 = FusionBlock(channel=self.n_features[2],height=128//8,width=256//8)
#         self.fuse4 = FusionBlock(channel=self.n_features[3],height=128//16,width=256//16)
        
        
        self.dLayer4 = ResConvBlock(self.n_features[4]+self.n_features[3]*2,self.n_features[3])
        self.dLayer3 = ResConvBlock(self.n_features[3]+self.n_features[2]*2,self.n_features[2])
        self.dLayer2 = ResConvBlock(self.n_features[2]+self.n_features[1]*2,self.n_features[1])
        self.dLayer1 = ResConvBlock(self.n_features[1]+self.n_features[0]*2,self.n_features[0])
        
        self.out = torch.nn.Conv2d(self.n_features[0],n_class,1)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, rgb, thr):
        thrStem = self.thrStem(thr)
        rgbStem = self.rgbStem(rgb)  # 256 
        
        
        thr = self.downsample(thrStem)
        rgb = self.downsample(rgbStem) # 128
#         thr,rgb = self.fuse1(thr,rgb)
        
        thrLayer1 = self.thrLayer1(thr)
        rgbLayer1 = self.rgbLayer1(rgb)
        
        
        thr = self.downsample(thrLayer1)
        rgb = self.downsample(rgbLayer1) # 64 
#         thr,rgb = self.fuse2(thr,rgb)
        
        thrLayer2 = self.thrLayer2(thr)
        rgbLayer2 = self.rgbLayer2(rgb)
        
        
        thr = self.downsample(thrLayer2)
        rgb = self.downsample(rgbLayer2) # 32 
#         thr,rgb = self.fuse3(thr,rgb)
        
        thrLayer3 = self.thrLayer3(thr)
        rgbLayer3 = self.rgbLayer3(rgb)
        
        
        thr = self.downsample(thrLayer3)
        rgb = self.downsample(rgbLayer3) # 16
#         thr,rgb = self.fuse4(thr,rgb)
        
        thrLayer4 = self.thrLayer4(thr)
        rgbLayer4 = self.rgbLayer4(rgb)

        sumLayer4 = self.bridge(torch.cat([rgbLayer4,thrLayer4],dim=1))
        
        
        x = self.upsample(sumLayer4)
        x = torch.cat([x,rgbLayer3,thrLayer3],dim=1)
        x = self.dLayer4(x)
        
        x = self.upsample(x)
        x = torch.cat([x,rgbLayer2,thrLayer2],dim=1)
        x = self.dLayer3(x)
        
        x = self.upsample(x)
        x = torch.cat([x,rgbLayer1,thrLayer1],dim=1)
        x = self.dLayer2(x)
        
        x = self.upsample(x)
        x = torch.cat([x,rgbStem,thrStem],dim=1)
        x = self.dLayer1(x)
        
        out = self.out(x)
        
        return out
        
