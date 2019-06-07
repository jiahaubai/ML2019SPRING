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

# use for strong baseline use with '0604_test_quan.pth'
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

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

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

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model2 = Net2().to(device)
state = torch.load('test_quan.pth')
model2.load_state_dict(state)

fp = open(sys.argv[2],'w', newline='') #'0604_test_quan.pth testing score 0.63666
w = csv.writer(fp)
title = ['id','label']
w.writerow(title)
model2.eval()
with torch.no_grad():
    for step, (b_x) in enumerate(test_loader):
     
        b_x = np.array([t.numpy() for t in b_x]).reshape(-1, 1, 48, 48)
        b_x = torch.from_numpy(b_x).to(device)
        output = model2(b_x)              # cnn output
        pred_y = torch.max(output, 1)[1].cpu().numpy()
        content = ['%d'%(step), pred_y[0]]
        w.writerow(content)
    fp.close()
print("finish write ans")