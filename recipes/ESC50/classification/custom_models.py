import torch.nn as nn
import torch.nn.functional as F


class weak_mxh64_1024(nn.Module):
    def __init__(self, nout):
        super(weak_mxh64_1024, self).__init__()

        glplfn = F.max_pool2d

        self.globalpool = glplfn
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer6 = nn.MaxPool2d(2)

        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer9 = nn.MaxPool2d(2)

        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer12 = nn.MaxPool2d(2)

        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer15 = nn.MaxPool2d(2)  #

        self.layer16 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer17 = nn.MaxPool2d(2)  #

        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, nout, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out1 = self.layer19(out)
        out = self.globalpool(out1, kernel_size=out1.size()[2:])
        out = out.view(out.size(0), -1)
        return out  # ,out1
