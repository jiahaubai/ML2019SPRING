import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as f
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import csv
import sys 

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 5, stride, padding = 2),
                nn.LeakyReLU(1./20),
                nn.BatchNorm2d(oup),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )

        def conv_dw(inp, oup, stride, padding = 1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, padding, groups=inp, bias=False),
                nn.LeakyReLU(1./20),
                nn.BatchNorm2d(inp),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.LeakyReLU(1./20),
                nn.BatchNorm2d(oup) 
            )
        
        self.model = nn.Sequential(
            conv_bn(1, 64, 1),
            conv_dw(64, 128, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            conv_dw(128,512, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
            conv_dw(512, 256, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
            nn.AvgPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256,7),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        
        return output

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(30, translate=(0.2,0.2), scale=(0.8, 1.2), shear=0.2, resample=False, fillcolor=0),
    #transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean = [0.5], std = [0.5])
])
valid_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Normalize(mean = [0.5], std = [0.5])
])

class trainset(Dataset):
    def __init__(self, data_tensor, target_tensor, loader=preprocess):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.loader = loader
        
    def __getitem__(self, index):
        fn = self.data_tensor[index]
        data = self.loader(fn)
        target = self.target_tensor[index]
        return data,target
        
    def __len__(self):
        return self.data_tensor.size()[0]

df = pd.read_csv(sys.argv[1],engine = 'python')
np_data = df.iloc[:,1].values.astype(str)
np_label = df.iloc[:,0].values

trainX = []
trainY = []
valX = []
valY = []

print('data reading')
for i in range(np_data.shape[0]):
    str_ = np_data[i].split(' ')
    tmp = list(np.array(str_).astype(int).reshape(1,48,48))
    if i%20 == 0:
        valX.append(tmp)
        valY.append(np_label[i])
    else:
        trainX.append(tmp)
        trainY.append(np_label[i])  # data augmentation
    
trainX = np.array(trainX).astype(float)
valX = np.array(valX).astype(float)
trainY = np.array(trainY)
valY = np.array(valY)

trainX = torch.from_numpy(trainX).float()# numpy轉torch
valX = torch.from_numpy(valX).float()# numpy轉torch
trainY = torch.from_numpy(trainY).long()
valY = torch.from_numpy(valY).long()
print(trainX[0])
print('finish')

BATCH_SIZE = 64
EPOCH = 500        
LR = 0.0001         

#把data set放入data loader裡面

train_data  = trainset(trainX,trainY)
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
val_data = trainset(valX,valY,loader=valid_preprocess)
val_loader = Data.DataLoader(dataset = val_data, batch_size = 1, shuffle = False)

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

cnn2 = Net2().to(device)
optimizer = torch.optim.Adam(cnn2.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(EPOCH):
    
    train_loss = 0
    correct = 0
    cnn2.train()  # Important: set training mode
    for step, (batch_x, batch_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()           # clear gradients for this training step
        output = cnn2(batch_x)           # cnn output
        loss = loss_func(output, batch_y)   # cross entropy loss
              
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients     
        
        train_loss += loss_func(output, batch_y).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(batch_y.view_as(pred)).sum().item()
    
    cnn2.eval()
    val_correct = 0
    
    with torch.no_grad():
        for (batch_valx, batch_valy) in val_loader:
            batch_valx, batch_valy = batch_valx.to(device), batch_valy.to(device)
            output = cnn2(batch_valx)
            pred = output.max(1, keepdim=True)[1]
            val_correct += pred.eq(batch_valy.view_as(pred)).sum().item()
    
    val_acc = val_correct/len(val_loader.dataset)
    if val_acc > best_acc :
        save_checkpoint('mobilenet.pth', cnn2)
        best_acc = val_acc
                              
    print('Epoch: ', epoch, '| train loss: %.6f' %(train_loss/len(train_loader.dataset)),\
          '| train acc: %.6f' %(correct/len(train_loader.dataset)),\
         '| val acc: %.6f'%(val_acc))