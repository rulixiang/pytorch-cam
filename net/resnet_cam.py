import torch.nn as nn
import torch.nn.functional as F
import torch 
from .resnet import resnet101, resnet50


class ResNet(nn.Module):

    def __init__(self, n_classes=20, backbone='resnet50'):
        super(ResNet, self).__init__()
        self.n_classes = n_classes

        if backbone == 'resnet50':
            self.resnet = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        elif backbone == 'resnet101':
            self.resnet = resnet101(pretrained=True, strides=(2, 2, 2, 1))

        self.stem = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.stage1 = nn.Sequential(self.resnet.layer1)
        self.stage2 = nn.Sequential(self.resnet.layer2)
        self.stage3 = nn.Sequential(self.resnet.layer3)
        self.stage4 = nn.Sequential(self.resnet.layer4)

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)

    def freeze_param(self):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        out = F.adaptive_avg_pool2d(x4, (1,1))
        out = self.classifier(out)
        out = out.view(-1, self.n_classes)

        cam = F.conv2d(x4, self.classifier.weight)
        cam = F.relu(cam)   
        #cam = torch.max(cam[0], cam[1].flip(-1))

        return out, cam

    def forward_cam(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        cam = F.conv2d(x4, self.classifier.weight)
        cam = F.relu(cam)

        return cam

if __name__ == "__main__":
    res50 = ResNet()
    dummy_input = torch.rand(4,3,321,321)
    out = res50(dummy_input)

    print(out.shape)