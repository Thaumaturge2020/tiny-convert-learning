import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

dT = 0.002

class MyDataset_load_differ(Dataset):
    def __init__(self,datapath):
        constractive_list = []
        for i in range(0,2):
            for j in range(0,10):
                for k in range(0,j):
                    I = i+1
                    J = j+1
                    K = k+1
                    a = os.path.join(datapath,str(I),str(J)+".txt")
                    b = os.path.join(datapath,str(I),str(K)+".txt")
                    c = 0
                    # a = transforms.ToTensor(a)
                    # b = transforms.ToTensor(b)
                    # print(a)
                    constractive_list.append((a,b,c))

        
        for i in range(0,2):
            for j in range(0,j):
                for k in range(0,10,2):
                    for w in range(0,10,3):
                        I = i+1
                        J = j+1
                        K = k+1
                        W = w+1
                        a = os.path.join(datapath,str(I),str(K)+".txt")
                        b = os.path.join(datapath,str(J),str(W)+".txt")
                        c = 1
                        # a = transforms.ToTensor(a)
                        # b = transforms.ToTensor(b)
                        constractive_list.append((a,b,c))

        self.constractive_list = constractive_list

        self.given_length = 10000
    
    def __len__(self):
        return len(self.constractive_list)
    
    def __getitem__(self, index):
        a,b,c = self.constractive_list[index]
        a = np.loadtxt(a)
        b = np.loadtxt(b)
        c = c
        a = torch.tensor(a)

        while(a.shape[1] < self.given_length):
            a = a.repeat(1,2)
        a = a[:,0:self.given_length]

        b = torch.tensor(b)
        
        while(b.shape[1] < self.given_length):
            b = b.repeat(1,2)
        b = b[:,0:self.given_length]
        
        c = c

        a = a.float()
        b = b.float()

        return a,b,c
    
