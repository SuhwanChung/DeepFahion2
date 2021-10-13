import os
import pandas as pd
import torch
import numpy as nptest

import cv2
from torch.utils.data import Dataset

from torchvision import transforms as T

from toolz import *
from toolz.curried import *



class FashionDataset(Dataset):
    def __init__(self, dataPath, splitPath,
                 test = True, 
                 aug   = identity):
        
        self.pts = parse(dataPath, splitPath, test)
        self.aug = aug

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, i):
        
        imgPath = self.pts[i]
        image = self.aug(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB))
        return (torch.tensor(image).to(torch.float32))
    


def Aug(test) :
    def inner(img):
        if test:            
            return T.Compose([T.ToPILImage(),
                              T.Resize((224, 224)),
                              T.CenterCrop(size=224),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img)
                    
        else:
            return T.Compose([T.ToPILImage(),
                              T.Resize((224, 224)),
                              T.CenterCrop(size=224),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img)
    return inner  


    
def parse(dataPath, splitPath, test):

    test = "test" 
    imgPaths = pd.read_csv(f"{splitPath}/{test}.txt", delimiter = " ", header = None)[0].tolist()
#   labels   = pd.read_csv(f"{splitPath}/{train}_attr.txt", delimiter = " ", header = None).values.tolist()        
    imgPaths = map(lambda x : f"{dataPath}/{x}")(imgPaths)
#   labels   = map(np.array)(labels)
    return list(imgPaths)                                               


'''
if __name__ == "__main__":
                
    imgPath   = "./Dataset"
    splitPath = "./Dataset/split"
    test     = True
    
    dataset = FashionDataset(imgPath, splitPath, test, Aug(test))
    
    image = daetaset.__getitem__(1)
'''