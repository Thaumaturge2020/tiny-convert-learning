import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import sys

from dataset_loader import MyDataset_load_differ

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6,64,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2,padding=1),
            nn.Conv1d(64,128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2,padding=1),
            nn.Linear(2500,5)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*5,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128)
        )

    def forward(self,x1,x2):

        # print(x1.shape)
        out1 = self.conv(x1)
        out1 = out1.view(out1.size(0), -1)
        # print(out1.shape)
        out1 = self.fc(out1)
        # print("test3")
        
        out2 = self.conv(x2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc(out2)

        return out1, out2
    
class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 1e-18

    def forward(self,output1,output2,label):
        dist = torch.sqrt(torch.sum(torch.pow(output1-output2,2),1)+self.eps)
        mdist = torch.clamp(self.margin - dist,min=0.0)
        loss = (1-label) * torch.pow(dist,2) + label * torch.pow(mdist,2)
        loss = torch.sum(loss) / 2.0 / output1.size()[0]
        return loss
    
    def accurate(self,output1,output2,label):
        dist = torch.sqrt(torch.sum(torch.pow(output1-output2,2),1)+self.eps)
        mdist = torch.clamp(self.margin - dist,min=0.0)

        label_me = dist > self.margin

        label_id = label > 0.5
        
        label_acc = torch.logical_not(torch.logical_xor(label_me,label_id))

        acc = torch.sum(label_acc) / output1.size()[0]

        return acc
    
    def predict(self,output1,output2):
        dist = torch.sqrt(torch.sum(torch.pow(output1-output2,2),1)+self.eps)
        mdist = torch.clamp(self.margin - dist,min=0.0)

        label_me = dist > self.margin

        return label_me

datapath = "/home/red-sakura/GIT/deep_learning/phrase"

train_dataset = MyDataset_load_differ(datapath)
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=0)


    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

model = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1000

state_dict = torch.load('model_v6_partial_40_1000.pth')
model.load_state_dict(state_dict)

for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    total_len = 0
    for batch_idx, (a, b, label) in enumerate(train_dataloader):
        # print(batch_idx)
        a, b, label = a.to(device), b.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1,output2 = model(a,b)
        loss = criterion(output1,output2,label)
        loss.backward()
        optimizer.step()

        acc = criterion.accurate(output1,output2,label)

        if (batch_idx+1) % 10 ==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_dataloader), loss.item(),acc))

        total_loss += loss.item()*a.shape[0]
        total_acc += acc*a.shape[0]
        total_len += a.shape[0]
    
    print('Epoch [{}/{}], Total_Loss:{}'.format(epoch+1,num_epochs,total_loss/total_len))
    print('Epoch [{}/{}], Total_Acc:{}'.format(epoch+1,num_epochs,total_acc/total_len))
            
    if epoch % 5 == 0:
        torch.save(model.state_dict(),'./model_v11_partial_'+str(epoch)+'_1000.pth')