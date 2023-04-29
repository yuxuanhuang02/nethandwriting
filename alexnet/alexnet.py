from torch import nn
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
class ConvLinear(nn.Module):
    def __init__(self,in_channels,out_channels,kernelsize,stride,padding,bias=False) -> None:
        super(ConvLinear,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernelsize,stride,padding,bias=bias),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU())
    def forward(self,x):
        return self.conv(x)

class Overlapping(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(Overlapping,self).__init__()
        self.maxpool = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,2),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU())
    def forward(self,x):
        return self.maxpool(x)

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super(AlexNet,self).__init__()
        self.conv1 = ConvLinear(3,96,11,4,0)
        self.pool1 = Overlapping(96,96)
        self.conv2 = ConvLinear(96,256,5,1,2)
        self.pool2 = Overlapping(256,256)
        self.conv3 = ConvLinear(256,384,3,1,1)
        self.conv4 = ConvLinear(384,384,3,1,1)
        self.conv5 = ConvLinear(384,256,3,1,1)
        self.pool3 = Overlapping(256,256)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.pool3(x7)
        return x8
net = AlexNet()
net = net.to(device)
x = torch.randn(1,3,227,227)
x = x.to(device)
print(net(x).shape)