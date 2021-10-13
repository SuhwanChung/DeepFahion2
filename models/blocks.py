from torchvision import models as models
import torch.nn as nn

def Encoder(name):
    return {"ResNet"     : lambda : (2048, nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])),
            "AlexNet"    : lambda : (256,  models.alexnet(pretrained=True).features),
            "VGGNet"     : lambda : (512,  models.vgg16(pretrained=True).features)}[name]()

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),            
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), # Drop out 60% chance
            nn.Linear(hidden_size, projection_size, bias=True),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.net(x)
