import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA

from PIL import Image
import pandas as pd
import numpy as np
import os
import csv
import glob
import sys

transform = transforms.Compose([
    transforms.ToTensor()
])

class dataset(data.Dataset):
    def __init__(self,img_root):
        self.img_root = img_root
    
    def __getitem__(self,index):
        img = Image.open(self.img_root[index])
        img = transform(img)
        
        return img
    
    def __len__(self):
        return len(self.img_root)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

intermediate_size = 1000
hidden_size = 100

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
                     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                     nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Linear(16 * 16 * 256, intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 16 * 16 * 256)
        self.deconv1 = nn.Sequential(
                       nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
                       nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(64)
        )
        self.deconv3 = nn.Sequential(
                       nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
                       nn.BatchNorm2d(32)
        )
        self.conv5 = nn.Sequential(
                     nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(3)
        )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 256, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
####################################################################################

input_image_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

print('load image')
#fnames = glob.glob("./images/images/*.jpg")
fnames = glob.glob(os.path.join(input_image_path, '*.' +'jpg'))

trainset = dataset(fnames)
trainLoader = data.DataLoader(trainset, batch_size = 128, shuffle = False)

print('load model')
model_1 = VAE().cuda()
load_checkpoint('autoencoder_epoch100.pth', model_1)

print('collect feature')
collect_features = []
model_1.eval()
with torch.no_grad():
    for idx, b_x in enumerate(trainLoader):
        if idx%100 == 0: print(idx)
        b_x = b_x.cuda()
        _, _, _,feature  = model_1(b_x)
        collect_features.extend(list(torch.squeeze(feature,-1).cpu().data.numpy()))
    collect_features = np.array(collect_features)    
print(collect_features.shape)

print('fit pca & kmeans')
#pca = PCA(n_components=100, copy=False, whiten=True, svd_solver='full')
#joblib.dump(pca,'pca.pkl')
# pca = joblib.load('pca.pkl')
# latent_vec = pca.fit_transform(collect_features)
latent_vec = np.load('latent_vec.npy')

kmeans = joblib.load('kmeans_1.pkl')
pred_result = kmeans.predict(latent_vec)

print('write answer')
#df = pd.read_csv('test_case.csv')
df = pd.read_csv(test_path)
imgName_1 = df['image1_name']
imgName_2 = df['image2_name']

print(len(imgName_1))
#fp = open('output_autoencoder.csv','w', newline='')
fp = open(output_path,'w', newline='')
w = csv.writer(fp)
title = ['id','label']
w.writerow(title)

for idx in range(len(imgName_1)): #len(imgName_1)
    
    if idx%10000 == 0: print(idx)

    if pred_result[imgName_1[idx]-1] == pred_result[imgName_2[idx]-1]:
        ans = 1
    elif pred_result[imgName_1[idx]-1] != pred_result[imgName_2[idx]-1]: 
        ans = 0
    
    content = [idx, ans]
    w.writerow(content)
fp.close()