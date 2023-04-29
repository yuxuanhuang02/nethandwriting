from torch import nn

import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class ConvLinear(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(ConvLinear,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class Downsampling(nn.Module):
    def __init__(self,channels) -> None:
        super(Downsampling,self).__init__()
        self.down = nn.Sequential(nn.Conv2d(channels,channels,3,2,1,bias=False),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU())
    def forward(self,x):
        return self.down(x)

class Upsampling(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        super(Upsampling,self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU())
    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode="nearest")
        out = self.up(up)
        return torch.cat((out,feature_map),dim=1)

class FCNNet(nn.Module):
    def __init__(self) -> None:
        super(FCNNet,self).__init__()
        self.conv1 = ConvLinear(3,64)
        self.down1 = Downsampling(64)
        self.conv2 = ConvLinear(64,128)
        self.down2 = Downsampling(128)
        self.conv3 = ConvLinear(128,256)
        self.down3 = Downsampling(256)
        self.conv4 = ConvLinear(256,512)
        self.down4 = Downsampling(512)
        self.conv5 = ConvLinear(512,1024)
        self.up1 = Upsampling(1024,512)
        self.conv6 = ConvLinear(1024,512)
        self.up2 = Upsampling(512,256)
        self.conv7 = ConvLinear(512,256)
        self.up3 = Upsampling(256,128)
        self.conv8 = ConvLinear(256,128)
        self.up4 = Upsampling(128,64)
        self.conv9 = ConvLinear(128,64)
        self.out = nn.Conv2d(64,3,1,1,0,bias=False)
        self.relu = nn.ReLU()



    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.conv2(x2)
        x4 = self.down2(x3)
        x5 = self.conv3(x4)
        x6 = self.down3(x5)
        x7 = self.conv4(x6)
        x8 = self.down4(x7)
        x9 = self.conv5(x8)
        x10 = self.up1(x9,x7)
        x11 = self.conv6(x10)
        x12 = self.up2(x11,x5)
        x13 = self.conv7(x12)
        x14 = self.up3(x13,x3)
        x15 = self.conv8(x14)
        x16 = self.up4(x15,x1)
        x17 = self.conv9(x16)
        return self.relu(self.out(x17))

fcn = FCNNet()
fcn = fcn.to(device)
x = torch.randn(2,3,256,256)
x = x.to(device)
print(fcn(x).shape)


