import torch
import torch.nn
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class ConvLinear(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False) -> None:
        super(ConvLinear,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.conv(x)

class Downsampling(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(Downsampling,self).__init__()
        self.down = ConvLinear(in_channels,out_channels,3,2,1,bias=False)
    def forward(self,x):
        return self.down(x)

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(Residual,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
            nn.Conv2d(out_channels,in_channels,3,1,1,bias=False)
            )
    def forward(self,x):
        return self.residual(x)+x

class darknet53(nn.Module):

    def __init__(self) -> None:
        super(darknet53,self).__init__()
        self.darknet = nn.Sequential(
            ConvLinear(3,32,3,1,1),
            Downsampling(32,64),
            Residual(64,32),
            Downsampling(64,128),
            Residual(128,64),
            Residual(128,64),
            Downsampling(128,256),
            Residual(256,128),
            Residual(256, 128),
            Residual(256, 128),
            Residual(256, 128),
            Residual(256, 128),
            Residual(256, 128),
            Residual(256, 128),
            Residual(256, 128),
            Downsampling(256,512),
            Residual(512,256),
            Residual(512, 256),
            Residual(512, 256),
            Residual(512, 256),
            Residual(512, 256),
            Residual(512, 256),
            Residual(512, 256),
            Residual(512, 256),
            Downsampling(512,1024),
            Residual(1024,512),
            Residual(1024, 512),
            Residual(1024, 512),
            Residual(1024, 512)
            )
    def forward(self,x):
        return self.darknet(x)

dark = darknet53()
dark = dark.to(device)
y = torch.randn((1,3,32,32))
y = y.to(device)
y = dark(y)
print(y.shape[0])
print(y.shape[2])