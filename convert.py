import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

dT = 0.002

datapath = "/home/red-sakura/GIT/deep_learning/phrase"

given_length=10000

for i in range(0,10):
    for j in range(0,10):
            I = i+1
            J = j+1

            print(I,i)

            a_index = os.path.join(datapath,str(I),str(J)+".txt")
            a_convert_index = os.path.join(datapath,str(I),str(J)+"_s.txt")
    
            a = np.loadtxt(a_index)
            a = torch.tensor(a)

            while(a.shape[1] + a.shape[1] < given_length):
                a = torch.nn.functional.pad(a,(0,a.shape[1]),mode="circular")
            a = torch.nn.functional.pad(a,(0,given_length-a.shape[1]),mode="circular")

            a_v = []
            for x in range(0,6):
                a_v.append([])
                now = 0.0
                for y in range(given_length):
                    a_v[x].append(now)
                    now += dT * a[x][y].item()
            a_s = []
            for x in range(0,6):
                a_s.append([])
                now = 0.0
                for y in range(given_length):
                    a_s[x].append(now)
                    now += dT * a_v[x][y]
                
            a = torch.tensor(a_s)


            a = a.float()

            np.savetxt(a_convert_index,a)
    
