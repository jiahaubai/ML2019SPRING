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

# transform data (training data, validation data, testing data)
test_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class testset(Dataset):
    def __init__(self, data_tensor, loader=test_preprocess):
        self.data_tensor = data_tensor
        self.loader = loader
        
    def __getitem__(self, index):
        fn = self.data_tensor[index]
        data = self.loader(fn)
        return data
        
    def __len__(self):
        return self.data_tensor.size()[0]

# model declare
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(1,48,48)
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.LeakyReLU(1./20),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(  # input shape(64,24,24)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(1./20),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)  
        )
        self.conv3 = nn.Sequential(  # input shape(128,12,12)
            nn.Conv2d(128, 512, 3, 1, 1),
            nn.LeakyReLU(1./20),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
        )
        self.conv4 = nn.Sequential(  # input shape(512,6,6)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(1./20),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
        )

        # fully connected layer
        self.flat = nn.Sequential(  # input shape (512, 3, 3)
            nn.Linear(3*3*512, 512 ),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 7)
        )
    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) 
        output = self.flat(x)
        
        return output

# load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

# reading test data
print('test data reading')
df = pd.read_csv(sys.argv[1] ,engine = 'python')
test_data = df.iloc[:,1].values.astype(str)
testX = []

for i in range(test_data.shape[0]):
    str_ = test_data[i].split(' ')
    tmp = list(np.array(str_).astype(int).reshape(1,48,48))
    testX.append(tmp)
    
testX = np.array(testX).astype(float)
testX = torch.from_numpy(testX).float()# numpy轉torch

print('finish reading test data')

test_data = testset(testX)
test_loader = Data.DataLoader(dataset = test_data, batch_size = 1, shuffle = False)
        
LR = 0.0001         
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
load_checkpoint('best_mode.pth', model, optimizer)

# write ans
fp = open(sys.argv[2],'w', newline='')
w = csv.writer(fp)
title = ['id','label']
w.writerow(title)
model.eval()
with torch.no_grad():
    for step, (b_x) in enumerate(test_loader):
     
        b_x = np.array([t.numpy() for t in b_x]).reshape(-1, 1, 48, 48)
        b_x = torch.from_numpy(b_x).to(device)
        output = model(b_x)              # cnn output
        pred_y = torch.max(output, 1)[1].cpu().numpy()
        content = ['%d'%(step), pred_y[0]]
        w.writerow(content)
    fp.close()
print("finish write ans")


