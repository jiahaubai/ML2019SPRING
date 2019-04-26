import numpy as np
import pandas as pd
from PIL import Image
import os
import copy
import sys
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.models as models
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

# get root and label data from csv
img_root = sys.argv[1] + '/'#'./hw5_data/images'
create_img_root = sys.argv[2] + '/'#'./PIL'

ImgId = list(np.load('ImgId.npy'))
TrueLabel = list(np.load('TrueLabel.npy'))


# write the data into dataloader
# normalize the dataset (use mean & var  from imageNET)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
class trainset(data.Dataset):
    def __init__(self,root,ImgId,TrueLabel,transform=transform):
        self.root = img_root
        self.fnames = ImgId
        self.labels = TrueLabel
        self.transform = transform
    
    def __getitem__(self,index):
        
        img_name = '%03d.png'%(self.fnames[index])
        img = Image.open(os.path.join(self.root+img_name))
        img = self.transform(img)
        target = torch.tensor(TrueLabel[index]).long()
        return img,target
    
    def __len__(self):
        return len(self.labels)

# test 'if get the data'
train_data  = trainset(img_root,ImgId,TrueLabel)
print('len:',train_data.__len__())
trainloader = data.DataLoader(train_data, batch_size=1,shuffle=False)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images), type(labels))
print(images.size(), labels.size())


# function: recreate the image from array 
def recreate_image(im_as_var):

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    #recreated_im = recreated_im[..., ::-1]
    return recreated_im


# prepare to use the GPU
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# load pretrained mode
model = models.resnet50(pretrained=True).to(device)
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

epsilon = 0.01
iter_num = 3
origin_correct = 0
after_attack_correct = 0
sum_L_inf = 0

for step,(image, target) in enumerate(trainloader):
    
    if step%10 == 0: print(step)
    
    image, target = image.to(device), target.to(device)
    image.requires_grad = True
    
    # use FGSM many times
    for i in range(iter_num):
        
        zero_gradients(image)
        output = model(image)
        loss = criterion(output, target)
        loss.backward() 
        adv_noise = epsilon * torch.sign(image.grad.data)
        image.data = image.data + adv_noise
    
    recreated_image = recreate_image(image.cpu())
    perturbed_image = image.data
    
    img = Image.fromarray(recreated_image)#.convert('RGB')
    img.save(create_img_root+"%03d.png"%(step))
    
    # sum the L-inf. norm
    sum_L_inf += torch.max(torch.abs(image - perturbed_image))
    
    # predict after attacking picture
    output = model(perturbed_image)
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    if final_pred.item() == target.item():
        after_attack_correct += 1
    
final_acc = after_attack_correct/float(len(trainloader))
average_L_inf = sum_L_inf
                                    
print("Epsilon:{} || Iter num:{} || After Attack Acc:{} || L-inf:{}".format(epsilon, iter_num, final_acc, average_L_inf))
