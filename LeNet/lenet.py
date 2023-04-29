from torch import nn
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
class ConvLinear(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False) -> None:
        super(ConvLinear,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
    def forward(self,x):
        return self.conv(x)

class avgpool(nn.Module):
    def __init__(self) -> None:
        super(avgpool,self).__init__()
        self.pool = nn.Sequential(nn.AvgPool2d(2,2),
                                  nn.Tanh())
    def forward(self,x):
        return self.pool(x)

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5,self).__init__()
        self.conv1 = ConvLinear(1,6,5,1,0)
        self.pool1 = avgpool()
        self.conv2 = ConvLinear(6,16,5,1,0)
        self.pool2 = avgpool()
        self.fc1 = nn.Linear(400,120,bias=False)
        self.fc2 = nn.Linear(120,84,bias=False)
        self.out = nn.Softmax(dim=1)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x4 = x4.view(-1,5*5*16)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        return self.out(x6)
net = LeNet5()
net = net.to(device)
x = torch.randn(1,1,32,32)
x = x.to(device)
print(net(x).shape)