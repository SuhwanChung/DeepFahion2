import os
from tqdm import tqdm
import numpy as np

from argparse import ArgumentParser
from easydict import EasyDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 

from models.fashionNet import FashionNet
from Data.generator import FashionDataset
from Data.augment import Aug

from utils2 import focalLoss

def parse():

    parser = ArgumentParser()

    parser.add_argument("--dataPath", type=str, default="./data/dataset")
    parser.add_argument("--splitPath", type=str, default="./data/dataset/split")
    parser.add_argument("--ckptPath", type=str, default="./ckpt")
    
    parser.add_argument("--encoderName", type=str, default = "ResNet", help = "AlexNet | VGGNet | ResNet")        
    parser.add_argument("--lr", type=float, default= 1e-5)    
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--gpuN", type=str, default="0", help = "0|1|2|3")
    parser.add_argument("--cpuN", type=int, default=6)
    parser.add_argument("--epochN", type=int, default=20)
    
    config = parser.parse_known_args()[0]
    
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuN
    
    return config

def trainStep(x, y, net, optimizer):

    weights = [[0.85, 0.83, 0.93, 0.92, 0.98, 0.52, 0.98],
               [0.7, 0.83, 0.48],
               [0.9, 0.81, 0.29],
               [0.58, 0.83, 1.0, 0.59],
               [0.94, 0.86, 0.32, 0.98, 0.99, 0.91],
               [0.85, 0.94, 0.21]]
    
    optimizer.zero_grad()
    
    logits = net(x)
    labels = y.permute(1,0)
    losses = [] 
    for logit, label, weight in zip(logits, labels, weights) :
        _loss = focalLoss(logit, label, weight = torch.tensor(weight).cuda())
        losses.append(_loss)
    loss = sum(losses)

    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def validStep(x, y, net):
    
    weights = [[0.85, 0.83, 0.93, 0.92, 0.98, 0.52, 0.98],
               [0.7, 0.83, 0.48],
               [0.9, 0.81, 0.29],
               [0.58, 0.83, 1.0, 0.59],
               [0.94, 0.86, 0.32, 0.98, 0.99, 0.91],
               [0.85, 0.94, 0.21]]
    
    logits = net(x)        
    labels = y.permute(1,0)
    
    losses = []
    preds  = []
    for logit, label, weight in zip(logits, labels, weights) :
        _loss = focalLoss(logit, label, weight = torch.tensor(weight).cuda())
        preds.append(torch.argmax(logit, -1))
        losses.append(_loss)
    
    loss = sum(losses)
    
    return loss, torch.stack(preds), labels


def train(trainDataLoader, validDataLoader, net, optimizer, config) :
    
    minLoss = 999
    
    net.train()    
    for e in range(config.epochN):
        
        print("TRAINING...")
        ######################################################
        losses = []
        net.train()        
        print(f"epoch: {e}/{config.epochN}")
        for (x, y, *_) in trainDataLoader:
            
            loss = trainStep(x.squeeze(0).cuda(),
                             y.cuda(),
                             net,
                             optimizer)
            
            losses.append(loss.item())                        
        losses = np.stack(losses)
        loss = float(str(np.mean(losses))[:4])
        print(f"TRAIN LOSS : {loss}")
        ######################################################
        
        print("\nVALIDATING...")
        ######################################################
        losses  = []
        preds   = []
        targets = []
        net.eval()
        for (x, y, *_) in validDataLoader:
            
            loss, pred, target = validStep(x.squeeze(0).cuda(),
                                           y.cuda(),
                                           net)
            
            preds.append(torch.flatten(pred).detach().cpu().numpy())
            targets.append(torch.flatten(target).detach().cpu().numpy())
            losses.append(loss.item())
            
        print(preds[-1])
        losses = np.stack(losses)
        acc    = sum(np.concatenate(preds) == np.concatenate(targets)) / len(np.concatenate(targets))
        
        loss = float(str(np.mean(losses))[:4])
        acc  = float(str(acc)[:4])
        print(f"VALID LOSS : {loss}")
        print(f"VALID ACC  : {acc * 100}")
    
        if loss < minLoss :
            print(f"(saving) the current loss {loss} is smaller than the previous loss {minLoss}")    
            
            directory = f"{config.ckptPath}"
            os.makedirs(directory, exist_ok=True)
            torch.save(net.state_dict(), f"{directory}/weights.pt")
            minLoss = loss
            
        else :
            print(f"(aborting) the current loss {loss} is bigger than the previous loss {minLoss}")
            
            
if __name__ == "__main__" :
    
    config = parse()
    
    # model, optmizer
    ###################################################    
    net    = FashionNet(config.encoderName).cuda()
    opt    = optim.Adam(net.parameters(), lr=config.lr)
    
    # dataloaders
    ###################################################
    trainDataLoader = DataLoader(FashionDataset(config.dataPath, config.splitPath,
                                                train = True, 
                                                aug   = Aug(train = True)),
                                 batch_size = config.batchSize,
                                 shuffle    = True)
    
    validDataLoader = DataLoader(FashionDataset(config.dataPath, config.splitPath,
                                                train = False, 
                                                aug   = Aug(train = False)),
                                 batch_size = config.batchSize,
                                 shuffle    = False)
    
    #_trainDataLoader = iter(trainDataLoader)
        
    train(trainDataLoader, validDataLoader, net, opt, config)