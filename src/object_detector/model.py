import torch
import torch.nn as nn

from model_utils import Bottleneck, Conv, C2F, SPPF

class Detect(nn.Module):
    def __init__(self, in_channels, out_channels, nc, reg_max):
        super(Detect, self).__init__()

        self.detect1 = nn.Sequential(
            Conv(in_channels, out_channels, padding=1),
            Conv(in_channels, out_channels, padding=1),
            nn.Conv2d(in_channels, 4*reg_max, kernel_size=1, stride=1, padding=0)
        )

        self.detect2 = nn.Sequential(
            Conv(in_channels, out_channels, padding=1),
            Conv(in_channels, out_channels, padding=1),
            nn.Conv2d(in_channels, nc, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.detect1(x), self.detect2(x)

class config:
    w = 0.25
    d = 0.33
    r = 2.0

class YOLOv8Backbone(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.conv0 = Conv(in_channels=3, out_channels=64*config.w, stride=2, padding=1)
        self.conv1 = Conv(in_channels=64, out_channels=128*config.w, stride=2, padding=1)
        self.C2f2 = C2F(shortcut=True, n=3*config.d)
        self.conv3 = Conv(in_channels=128, out_channels=256*config.w, stride=2, padding=1)
        self.C2f4 = C2F(shortcut=True, n=6*config.d)
        self.conv5 = Conv(in_channels=256, out_channels=512*config.w, stride=2, padding=1)
        self.C2f6 = C2F(shortcut=True, n=6*config.d)
        self.conv7 = Conv(in_channels=512, out_channels=512*config.w*config.r, stride=2, padding=1)
        self.C2f8 = C2F(shortcut=True, n=3*config.d)


        self.sppf = SPPF()
    
    def forward(self, x):
        c2f2 = self.C2f2(self.conv1(self.conv0(x)))
        c2f4 = self.C2f4(self.conv3(c2f2))
        c2f6 = self.C2f6(self.conv5(c2f4))
        sppf = self.sppf(self.C2f8(self.conv7(c2f6)))

        return c2f4, c2f6, sppf

class YOLOv8(nn.Module):
    def __init__(self):
        super.__init__()