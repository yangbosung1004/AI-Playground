import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

class vgg16(nn.Module):
    def __init__(self, num_classes):
        super(vgg16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.fc_layer = nn.Linear(1000,1000)
    def forward(img):
        x = self.vgg16(img)
        y = self.fc_layer(x)
        return y

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.fc_layer = nn.Linear(1000,1000)

    def forward(img):
        x = self.resnet50(img)
        y = self.fc_layer(x)
        return y
