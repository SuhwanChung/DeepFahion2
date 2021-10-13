import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models.fashionNet import FashionNet
from Data.test_generator import FashionDataset, Aug

dataPath = "./Data/dataset"
splitPath = "./Data/dataset/split"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testDataLoader = DataLoader(FashionDataset(dataPath, splitPath, test = True, aug = Aug(test = True)),
                             batch_size = 1, # Check the batch size for test
                             shuffle    = False)

model = FashionNet("ResNet")
model.load_state_dict(torch.load("./ckpt/weights.pt"))
model.eval()

if torch.cuda.is_available():
    model.cuda()

pred_labels = []
y_pred_list = []
with torch.no_grad():
    for X_batch in testDataLoader:
        y_pred_tags = np.array([], dtype=int)
        # X_batch = X_batch.squeeze(0)
#        print(X_batch.shape)
        X_batch = X_batch.to(device)
        # print(X_batch.shape)
        y_test_pred = model(X_batch)
        # y_test_pred = np.array(y_test_pred)
        # print(y_test_pred.shape)
        # y_pred_tags = np.argmax(y_test_pred, axis = 1)
        for y_test_pred_tensor in y_test_pred:
#            print(y_test_pred_tensor.shape)
            y_test_pred_tensor = y_test_pred_tensor.cpu()
            y_pred_tag = torch.argmax(y_test_pred_tensor, dim = 1).numpy()
#            print(y_pred_tag)
            y_pred_tags = np.append(y_pred_tags, y_pred_tag)
        y_pred_list.append(y_pred_tags)
        
#print(y_pred_list)
pred_labels = [a.squeeze().tolist() for a in y_pred_list]
#print(pred_labels)

with open('prediction.txt', 'w') as txt_file:
    for line in pred_labels:
        txt_file.write(" ".join(map(str, line)) + "\n")