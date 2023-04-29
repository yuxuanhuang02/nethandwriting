import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
class Conv64(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        super(Conv64,self).__init__()
        self.conv64 =nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self,x):
        return self.conv64(x)

class Conv128(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        super(Conv128,self).__init__()
        self.conv128 =nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self,x):
        return self.conv128(x)

class Conv256(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        super(Conv256,self).__init__()
        self.conv256 =nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,in_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self,x):
        return self.conv256(x)

class Conv512(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        super(Conv512,self).__init__()
        self.conv512 =nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,in_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels,out_channels,3,1,padding="same",bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self,x):
        return self.conv512(x)
class Maxpool(nn.Module):
    def __init__(self) -> None:
        super(Maxpool,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        return self.pool(x)

class VGG16(nn.Module):
    def __init__(self) -> None:
        super(VGG16,self).__init__()
        self.conv1 = Conv64(3,64)
        self.pool = Maxpool()
        self.conv2 = Conv128(64,128)
        self.conv3 = Conv256(128,256)
        self.conv4 = Conv512(256,512)
        self.conv5 = Conv512(512,512)
        self.fc1 = nn.Linear(7*7*512,1*4096,bias=False)
        self.fc2 = nn.Linear(1*4096,1*4096,bias=False)
        self.fc3 = nn.Linear(1*4096,1*1000,bias=False)
        self.out = nn.Softmax(dim=1)
    def forward(self,x):
        x1 = self.conv1(x)

        x2= self.pool(x1)
        x3 = self.conv2(x2)

        x4 = self.pool(x3)
        x5 = self.conv3(x4)

        x6 = self.pool(x5)
        x7 = self.conv4(x6)

        x8 = self.pool(x7)
        x9 = self.conv5(x8)

        x10 = self.pool(x9)
        x10 = x10.view(-1,7*7*512)
        x11 = self.fc1(x10)
        x12 = self.fc2(x11)
        x13 = self.fc3(x12)
        x14 = self.out(x13)
        return x14
vggnet = VGG16()
vggnet.to(device)
x = torch.randn(1,3,224,224)
x = x.to(device)
y = vggnet(x)
print(y.shape)
print(vggnet)