import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import math
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from dataset_loader import MyDataset_load_differ

from PIL import Image

size_image = 512

#这段代码的意图：
#我要用一张图片拟合出一个多项式的系数，这个多项式是r关于theta的函数
#thera = f(r)，f(r)现在是多项式系数
#distance = min{sqrt((r*cos(theta)-x)^2 + (r*sin(theta)-y)^2}，r从0到size_image，当极坐标系下的(theta,r)触碰到障碍无法继续前进时，停止计算distance，返回算出的最小distance
#这个distance是我要拟合的目标，我要让这个distance最小
#求导梯度现在为0,参数反向传播无法拟合
#写完这个去玩玩OS

class CNNcoefNetwork(nn.Module):
    def __init__(self):
        super(CNNcoefNetwork,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.Conv2d(64,128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(128*128,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,4)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,8)
        )

    def forward(self,x1):
        out1 = self.conv1(x1)
        out1 = out1.view(out1.shape[0],-1)
        out1 = self.conv2(out1)
        out1 = out1.view(-1)
        # print(out1.shape)
        out1 = self.fc(out1)

        return out1

device_cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class EuclidLoss(nn.Module):
    def __init__(self,margin=1.0,start_point = torch.tensor([0]),end_point = torch.tensor([0]),input_image = np.array([]),coef = 0):
        super(EuclidLoss,self).__init__()
        self.margin = margin
        self.start_point = start_point
        self.end_point = end_point
        self.eps = 1e-18
        self.input_image = input_image
        self.coef = coef

    def forward(self,output):
        R_distance = torch.tensor(1e18,requires_grad=True).to(device)
        for r in range(size_image):
            theta = torch.tensor(0.0,requires_grad=True).to(device)
            for (label,element) in enumerate(output):
                theta = theta + pow(r,label)*element

            now_end_point = self.start_point + torch.tensor([r*torch.cos(theta),r*torch.sin(theta)],requires_grad=True).to(device)
            
            if R_distance.item() > torch.sqrt((now_end_point[0]-self.end_point[0])**2 + (now_end_point[1]-self.end_point[1])**2).item() :
                R_distance = torch.sqrt((now_end_point[0]-self.end_point[0])**2 + (now_end_point[1]-self.end_point[1])**2)

            if self.input_image[int(now_end_point[0].item())][int(now_end_point[1].item())] < 160:
                return R_distance

        
        return R_distance
    
    def get_distance(self,output):
        R_distance = 1e18
        for r in range(size_image):
            theta = torch.tensor(0.0).to(device)
            for (label,element) in enumerate(output):
                theta += pow(r,label)*element


            now_end_point = self.start_point + torch.tensor([r*torch.cos(theta),r*torch.sin(theta)]).to(device)

            R_distance = min(R_distance,torch.sqrt((now_end_point[0]-self.end_point[0])**2 + (now_end_point[1]-self.end_point[1])**2))

            if self.input_image[int(now_end_point[0].item())][int(now_end_point[1].item())] < 160:
                return R_distance
        
        return R_distance

datapath = "/home/red-sakura/GIT/tiny-convert-learning/phrase"

train_dataset = MyDataset_load_differ(datapath)
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=0)


    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

model = CNNcoefNetwork()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

model.to(device)

num_epochs = 1000

# state_dict = torch.load('model_v6_partial_40_1000.pth')
# model.load_state_dict(state_dict)

for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    total_len = 0
    for batch_idx, (my_map, start_point, end_point) in enumerate(train_dataloader):
        # print(batch_idx)
        my_map = my_map.to(device)
        start_point = start_point.view(-1).to(device)
        end_point = end_point.view(-1).to(device)


        criterion = EuclidLoss(start_point=start_point,end_point=end_point,input_image=my_map[0],coef=0.1)
        
        optimizer.zero_grad()
        output = model(my_map)
        loss = criterion(output)
        loss.backward()
        print(model.parameters().grad)
        optimizer.step()

        target_distance = criterion.get_distance(output)

        if (batch_idx+1) % 10 ==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Target_Distance: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_dataloader), loss.item(),target_distance))

        total_loss += loss.item()*my_map.shape[0]
        total_acc += target_distance*my_map.shape[0]
        total_len += my_map.shape[0]
    
    print('Epoch [{}/{}], Total_Loss:{}'.format(epoch+1,num_epochs,total_loss/total_len))
    print('Epoch [{}/{}], Total_Acc:{}'.format(epoch+1,num_epochs,total_acc/total_len))
            
    if epoch % 5 == 0:
        torch.save(model.state_dict(),'./model_'+str(epoch)+'_1000.pth')