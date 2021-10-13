import torch
import torch.nn as nn
from .blocks import Encoder, MLP

class FashionNet(nn.Module):
    
    def __init__(self, encoderName):
        super().__init__() 
        
        self.hiddenSize, self.encoder = Encoder(encoderName)
        
        self.mlps = {"category1" : MLP(self.hiddenSize, 7),
                     "category2" : MLP(self.hiddenSize, 3),
                     "category3" : MLP(self.hiddenSize, 3),
                     "category4" : MLP(self.hiddenSize, 4),
                     "category5" : MLP(self.hiddenSize, 6),
                     "category6" : MLP(self.hiddenSize, 3)} ; self.mlps = nn.ModuleDict(self.mlps)
        
    def forward(self, img):
        
        z = self.encoder(img).squeeze()
        if z.dim() == 1:
            z = torch.unsqueeze(z, 0)
        
        logits = []        
        for k, mlp in self.mlps.items() :            
            logits.append(mlp(z))
            
        return logits
    
if __name__ == "__main__" :
    
    image = torch.randn(10,3,256,256).cuda()
    
    net = FashionNet("ResNet").cuda()
    
    logits = net(image)