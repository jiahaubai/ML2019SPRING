import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch
from lime import lime_image
from skimage.segmentation import slic
import time

start = time.time()

np.random.seed(0) #numpy
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
torch.backends.cudnn.deterministic=True # cudnn
torch.backends.cudnn.benchmark = False

path = sys.argv[2]
# load data
print('data reading')

df = pd.read_csv(sys.argv[1] ,engine = 'python')
np_data = df.iloc[:,1].values.astype(str)
np_label = df.iloc[:,0].values
trainX = []
trainY = []

for i in range(0,np_data.shape[0]):#np_data.shape[0]
    str_ = np_data[i].split(' ')
    tmp = list(np.array(str_).astype(int).reshape(1,48,48))
    
    trainX.append(tmp)
    trainY.append(np_label[i])

trainX = np.array(trainX).astype(float)
trainY = np.array(trainY)
trainX = torch.from_numpy(trainX).float()# numpy轉torch
trainY = torch.from_numpy(trainY).long()

print('reading finish')


# data loder
torch_dataset_train = torch.utils.data.TensorDataset(trainX,trainY)
train_loader = Data.DataLoader(dataset = torch_dataset_train, batch_size = 1 , shuffle = True)

# decalar model architectrue
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 48, 48)
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(  # input shape (64, 24, 24)
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(2, 2, 0)  
        )
        self.conv3 = nn.Sequential(  # input shape (64, 12, 12)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(  # input shape (128, 12, 12)
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)
        )
        self.conv5 = nn.Sequential(  # input shape (128, 6, 6)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
            
        )
        self.conv6 = nn.Sequential(  # input shape (256, 6, 6)
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )

        self.conv1.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
# using GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
LR = 0.001
cnn = CNN().to(device)
#print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# load pretrained model parameter
checkpoint = torch.load('mode-4.pth')
states_to_load = {}
for name, param in checkpoint['state_dict'].items():
    if name.startswith('conv'):
        states_to_load[name] = param

model_state = cnn.state_dict()
model_state.update(states_to_load)
        
cnn2 = CNN().to(device)
cnn2.load_state_dict(model_state)
########################## plot salienct_map ##############################
def compute_saliency_maps(x, y, model):
    
    np.random.seed(0) #numpy
    torch.manual_seed(0) # cpu
    torch.cuda.manual_seed(0) #gpu
    torch.backends.cudnn.deterministic=True # cudnn
    torch.backends.cudnn.benchmark = False
    
    model.eval()
    x.requires_grad_()
    y_pred = model(x.to(device))
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.to(device))
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency

def plot_image():
    
    pos = [0,388,2,14,3,15,4]
    for i in range(len(pos)):
        print(i,trainY[pos[i]])
        plt.figure(figsize=(10,10))
        idx = pos[i]
        x = torch.unsqueeze(trainX[idx],1)
        y = trainY[idx].reshape(1)
        
        x_org = x.squeeze().numpy()
        # Compute saliency maps for images in X
        saliency = compute_saliency_maps(x, y, cnn2)
        saliency = saliency.detach().cpu().numpy()
        #plt.subplot(1,2,1)
        #plt.imshow(x_org, cmap=plt.cm.gray)
        #plt.subplot(1,2,2)
        #plt.imshow(saliency, cmap=plt.cm.jet)
        plt.imsave(path+'./fig1_'+ str(i)+'.jpg', saliency, cmap=plt.cm.jet)
        #plt.show()


########################## end plot salienct_map ##############################

########################## plot gradient ascent ##############################
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gradient(x,filter_idx):
    
    np.random.seed(0) #numpy
    torch.manual_seed(0) # cpu
    torch.cuda.manual_seed(0) #gpu
    torch.backends.cudnn.deterministic=True # cudnn
    torch.backends.cudnn.benchmark = False
    
    cnn2.eval()
    x.requires_grad_()
    loss = torch.sum(cnn2.conv1(x)[:,filter_idx,:,:])
    loss.backward()
    return x.grad.squeeze().cpu().data.numpy(), loss.cpu().data.numpy()

def plot_gradient_ascent_trainingData():
    
    choose_img = torch.unsqueeze(trainX[0],1).data.numpy()
    lr = 0.05
    Epoch = 300
    
    plt.figure(figsize=(10,10))

    for filter_idx in range(64):
    
        input_img_data = choose_img/255
        
        for epoch in range(Epoch):
        
            x = torch.autograd.Variable(torch.from_numpy(input_img_data).float()).to(device)
            grad, loss = gradient(x,filter_idx)
        
            grad /= (np.sqrt(np.mean(np.square(grad))) + 1e-5)
            input_img_data += grad * lr
    
        #print(filter_idx,epoch, loss)    
        img = input_img_data[0]
        img = deprocess_image(img).reshape(48,48)
        plt.subplot(8,8, filter_idx+1)
        plt.imshow(img, cmap=plt.cm.jet)
        plt.axis('off')
    
    print('trainingData finish')
    plt.savefig(path+"./fig2_2.jpg")
    
def plot_gradient_ascent_random():
    
    lr = 0.05
    Epoch = 300
    
    plt.figure(figsize=(10,10))

    for filter_idx in range(64):
        
        np.random.seed(0) #numpy
        torch.manual_seed(0) # cpu
        torch.cuda.manual_seed(0) #gpu
        torch.backends.cudnn.deterministic=True # cudnn
        torch.backends.cudnn.benchmark = False
        
        input_img_data = (np.random.random((1, 1, 48, 48)) * 20 + 128.)/255
        
        for epoch in range(Epoch):
        
            x = torch.autograd.Variable(torch.from_numpy(input_img_data).float()).to(device)
            grad, loss = gradient(x,filter_idx)
        
            grad /= (np.sqrt(np.mean(np.square(grad))) + 1e-5)
            input_img_data += grad * lr
    
        #print(filter_idx,epoch, loss)    
        img = input_img_data[0]
        img = deprocess_image(img).reshape(48,48)
        plt.subplot(8,8, filter_idx+1)
        plt.imshow(img, cmap=plt.cm.jet)
        plt.axis('off')
    
    print('random start pic finish')
    plt.savefig(path+"./fig2_1.jpg")
########################## end plot gradient ascent ##############################

########################## plot lime ##############################
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def predict(input):
    
    np.random.seed(0) #numpy
    torch.manual_seed(0) # cpu
    torch.cuda.manual_seed(0) #gpu
    torch.backends.cudnn.deterministic=True # cudnn
    torch.backends.cudnn.benchmark = False
    
    r = input[:,:,:,0]
    gray = 1 * r 
    gray = torch.tensor(gray)
    gray = gray.view(10,1,48,48)
    
    cnn2.eval()   
    with torch.no_grad():
        x = gray.to(device)
        output = torch.squeeze(cnn(x))
    return softmax(output.cpu().data.numpy())

def segmentation(input):
    #Input: image numpy array
    #Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    return slic(input, n_segments=200)

def explain(img, predict):
    
    np.random.seed(0) #numpy
    torch.manual_seed(0) # cpu
    torch.cuda.manual_seed(0) #gpu
    torch.backends.cudnn.deterministic=True # cudnn
    torch.backends.cudnn.benchmark = False
    
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(
                            image=img,
                            top_labels=7,
                            classifier_fn=predict,
                            segmentation_fn=segmentation,
                            num_features=5,
                            batch_size=10
    )
    
    return explaination

def get_image(explaination,Label):
    
    np.random.seed(0)
    image, mask = explaination.get_image_and_mask(
                            label=Label,
                            positive_only=False,
                            hide_rest=False,
                            num_features=5,
                            min_weight=0.0
                    )

    # save the image
    plt.imsave(path+"./fig3_"+str(Label)+'.jpg', image)
    plt.imshow(image)

def lime_plot():
    pos = [0,388,2,14,3,15,4]
    for i in range(len(pos)):#
        print(i)
        x = trainX[pos[i]]/255
        img = x.view(48, 48, 1).expand( -1, -1, 3)
        img = img.data.numpy()
        explaination = explain(img, predict)
        get_image(explaination,i)


########################## end lime ##############################


print('plot salienct_map')
plot_image()
print('finisg plot salienct_map')

print('plot gradient ascent')
plot_gradient_ascent_trainingData()
plot_gradient_ascent_random()
print('finisg plot gradient ascent')

print('start plot lime')
lime_plot()
print('end plot lime')

end = time.time()

print('time:',end-start)