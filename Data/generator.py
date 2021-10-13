import os
import pandas as pd
import torch
import numpy as np

import cv2
from torch.utils.data import Dataset

from torchvision import transforms as T

from toolz import *
from toolz.curried import *

class FashionDataset(Dataset):
    
    def __init__(self, dataPath, splitPath,
                 train = True, 
                 aug   = identity):
        
        self.pts = parse(dataPath, splitPath, train)
        self.aug = aug

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, i):
        
        imgPath, label = self.pts[i]
        
        image = self.aug(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB))
       
        return (torch.tensor(image).to(torch.float32),
                torch.tensor(label).to(torch.long))
    
'''
def Aug(train) :
    
    def inner(img):
        
        if train:            
            return T.Compose([T.ToPILImage(),
                              T.Resize((224, 224)), # 256
                              T.RandomHorizontalFlip(p=0.5),
                              T.RandomRotation(degrees=45),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img) # Imagenet standard, Jacob dropped
                    
        else:
            return T.Compose([T.ToPILImage(),
                              T.Resize((224, 224)),
                              T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])(img) # Imagenet standard, Jacob dropped
        
    return inner
'''
    
def parse(dataPath, splitPath, train):
    
    train = "train" if train else "val"

    imgPaths = pd.read_csv(f"{splitPath}/{train}.txt", delimiter = " ", header = None)[0].tolist()
    labels   = pd.read_csv(f"{splitPath}/{train}_attr.txt", delimiter = " ", header = None).values.tolist()
        
    imgPaths = map(lambda x : f"{dataPath}/{x}")(imgPaths)
    labels   = map(np.array)(labels)
    
    return list(zip(imgPaths, labels))

if __name__ == "__main__":
                
    imgPath   = "./dataset"
    splitPath = "./dataset/split"
    train     = True
    
    daetaset = FashionDataset(imgPath, splitPath, train, Aug(train))
    
    image, label = daetaset.__getitem__(1)