import torch
import torch.nn as nn

class condig:
    h=10
    w=10
    c=10
    c_in = 10
    c_out = 10

class Detect(nn.Module):
    def __init__(self, in_channels, out_channels, numclasses, reg_max):
        super(Detect, self).__init__()

        self.detect1 = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(in_channels, out_channels),
            nn.Conv2d(in_channels, 4*reg_max, kernel_size=1, stride=1, padding=0)
        )

        self.detect2 = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(in_channels, out_channels),
            nn.Conv2d(in_channels, numclasses, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.detect1(x), self.detect2(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(num_features = out_channels),
            nn.SELU(inplace = True)
        )
    
    def forward(self, x):
        return self.convolution(x)

class SPPF(nn.Module): # Spatial Pyramid Pooling Fast
    def __init__(self, in_channels, out_channels, pool_kernel_size=5):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
        self.conv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
    

class C2F(nn.Module): # Cross‑Stage Partial “fused” block
    def __init__(self, in_channels, out_channels, num_blocks = 1, shortcut = False, expansion=0.5):
        super(C2F, self).__init__()

        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, 2*hidden_channels, kernel_size=1, padding=0)
        self.bottlenecks = nn.ModuleList([Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(num_blocks)])
        self.conv2 = Conv(2*hidden_channels + num_blocks*hidden_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        y = self.conv1(x)
        y1, y2 = y.chunk(2, dim=1)
        outputs = [y1]

        for bottleneck in self.bottlenecks:
            y1 = bottleneck(y1)
            outputs.append(y1)
        return self.conv2(torch.cat(outputs + [y2], dim=1))
    


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut:bool = True, expansion:int = 0.5):
        super(Bottleneck, self).__init__()

        self.add = shortcut and in_channels == out_channels
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, padding=1)
        self.conv2 = Conv(hidden_channels, out_channels, padding=1)

    def forward(self, x):
        conv = self.conv2(self.conv1(x))
        return x+conv if self.add else conv