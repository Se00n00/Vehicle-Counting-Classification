import torch
import torch.nn as nn

from model_utils import Bottleneck, Conv, C2F, SPPF

class config:
    w = 0.25
    d = 0.33
    r = 2.0

class YOLOv8Backbone(nn.Module):
    def __init__(self, config):
        super(YOLOv8Backbone, self).__init__()
        
        w,d,r = config.w, config.d, config.r
        self.conv0 = Conv(3, int(64*w), stride=2)
        self.conv1 = Conv(in_channels=int(64*w), out_channels=int(128*w), stride=2)
        self.C2f2 = C2F(int(128*w), int(128*w), shortcut=True, num_blocks=int(3*d))
        self.conv3 = Conv(in_channels=int(128*w), out_channels=int(256*w), stride=2)
        self.C2f4 = C2F(int(256*w), int(256*w), shortcut=True, num_blocks=int(6*d))
        self.conv5 = Conv(in_channels=int(256*w), out_channels=int(512*w), stride=2)
        self.C2f6 = C2F(int(512*w), int(512*w), shortcut=True, num_blocks=int(6*d))
        self.conv7 = Conv(in_channels=int(512*w), out_channels=int(512*w*r), stride=2)
        self.C2f8 = C2F(int(512*w*r), int(512*w*r), shortcut=True, num_blocks=int(3*d))


        self.sppf = SPPF(int(512*w*r), int(512*w*r))
    
    def forward(self, x):
        c2f2 = self.C2f2(self.conv1(self.conv0(x)))
        c2f4 = self.C2f4(self.conv3(c2f2))
        c2f6 = self.C2f6(self.conv5(c2f4))
        sppf = self.sppf(self.C2f8(self.conv7(c2f6)))

        return c2f4, c2f6, sppf

class YOLOv8Neck(nn.Module):
    def __init__(self, config):
        super(YOLOv8Neck, self).__init__()

        w,d,r = config.w, config.d, config.r
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f12 = C2F(int(512*w*(1+r)), int(512*w), num_blocks=int(3*d))
        self.c2f15 = C2F(int(768*w), int(256*w), num_blocks=int(3*d))

        self.conv16 = Conv(int(256*w), int(512*w), stride=2)
        self.c2f18 = C2F(int(1024*w), int(512*w), num_blocks=int(3*d))
        self.conv19 = Conv(int(512*w), int(512*w), stride=2)
        self.c2f21 = C2F(int(512*w*(1+r)), int(512*w*r), num_blocks=int(3*d))

    def forward(self, c2f4, c2f6, sppf):
        # Bottom To Up
        upsample_sppf = self.upsample(sppf)
        C2f12 = self.c2f12(torch.cat([c2f6, upsample_sppf], dim=1))
        upsample_c212 = self.upsample(C2f12)
        C2f15 = self.c2f15(torch.cat([c2f4, upsample_c212], dim=1))

        # Up To Down
        C2f18 = self.c2f18(torch.cat([C2f12, self.conv16(C2f15)], dim=1))
        C2f21 = self.c2f21(torch.cat([sppf, self.conv19(C2f18)], dim=1))

        return C2f15, C2f18, C2f21


class YOLOv8Head(nn.Module):
    def __init__(self, config):
        super(YOLOv8Head, self).__init__()


class YOLOv8(nn.Module):
    def __init__(self, config):
        super(YOLOv8, self).__init__()

        self.backbone = YOLOv8Backbone(config)
        self.neck = YOLOv8Neck(config)
    
    def forward(self, x):
        c2f4, c2f6, sppf = self.backbone(x)
        c2f15, c2f18, c2f21 = self.neck(c2f4, c2f6, sppf )