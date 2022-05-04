import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class UP(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(UP, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)

        self.conv3x31 = nn.Conv2d(low_in_plane, out_plane, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)
        self.conv1x11 = nn.Conv2d(low_in_plane, out_plane, 1)

        self.DLConv = DLConv(low_in_plane, out_plane, 2)

    def forward(self, high_x, low_x):
        low_x1=low_x

        high_x =self.conv3x3(high_x)
        high_x= self.conv1x11(high_x)
        low_x=self.conv3x31(low_x)
        low_x1=self.DLConv(low_x1)

        low_x = self.conv1x1(low_x+low_x1)

        return high_x +low_x


class DLConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DLConv, self).__init__(*modules)
class DLPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DLPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class DL(nn.Module):
    def __init__(self, in_channels,out_channels1, atrous_rates):
        super(DL, self).__init__()
        out_channels =out_channels1
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2,rate3 = tuple(atrous_rates)
        modules.append(DLConv(in_channels, out_channels, rate1))
        modules.append(DLConv(in_channels, out_channels, rate2))
        modules.append(DLConv(in_channels, out_channels, rate3))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Newblock(nn.Module):
    def __init__(self, in_channels,out_channels1):
        super(Newblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels1, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels1, 1)
        self.aspp=ASPP(in_channels,out_channels1,[1,2,3])

    def forward(self, l1):
        x1 = self.conv3x3(l1)
        x2 = self.conv1x1(l1)
        x3 = self.aspp(l1)

        return x2+x3+x1

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x)

        out = self.relu(o2 + x)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 64),
        )

        self.conv3 = nn.Sequential(
            Residual(64, 128, stride=2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
        )

        self.conv4 = nn.Sequential(
            Residual(128, 256, stride=2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
        )

        self.conv5 = nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512),
            Residual(512, 512),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.view((x.shape[0], -1))

        out = self.fc(out)

        return out

class LRSRModel(nn.Module):
    """
    downsample ratio=2
    """

    def __init__(self):
        super(LRSRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)




        self.dl1 = DL(6,12, [1,2,3])
        self.dl2 = DL(12, 24, [1,2,3])
        self.dl3 = DL(24, 48, [1,2,3])

        self.resnet = nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),

        )
        #


        self.seb1 = UP(48, 24, 24)
        self.seb2 = UP(24, 12, 12)
        self.seb3 = UP(12, 6, 6)

        self.map = nn.Conv2d(6, 1, 1)


    def forward(self, x):
        x1 = self.conv1(x)
        x1=self.resnet(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)



        m1=self.dl1(x1)

        m2 = self.dl2(m1)

        m3 = self.dl3(m2)

        up1 = self.seb1(m3, m2 )
        up2 = self.seb2(up1, m1)
        up3 = self.seb3(up2, x1)

        out = self.map(up3)
        return out

