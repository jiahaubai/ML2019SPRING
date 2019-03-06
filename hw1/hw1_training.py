import pandas as pd
import numpy as np
import sys

input_file = sys.argv[1]

df = pd.read_csv(input_file, engine = 'python').iloc[:,3:].values # hour starts from col 3

data = [[] for i in range(18)]
for row in range(len(df)):
    temp = list(df[row])
    for i in range(len(temp)):
        if temp[i] == 'NR':
            temp[i] = 0
        if float(temp[i]) < 0:
            temp[i] = 0
        temp[i] = float(temp[i])
        data[ row%18 ].append(temp[i])
        
data = np.array(data)

trainX = []
trainY = []
back_len = 9                             #<--- 5
start = 0                                #<--- 4
#(back_len,start) = (5,4) use 5 feature
#(back_len,start) = (9,0) use 10 feature
element_len = data.shape[1]
print(data.shape)


for start_day in range(0, element_len, 480):
    print(start_day,start_day+479)
    
    for day in range( start_day, start_day+480-back_len-start, 1 ):
        reshape_data = data[ :, day+start:day+start+back_len].reshape(162)
        trainX.append(reshape_data)
        trainY.append(data[9,day+start+back_len])
    
trainX = np.array(trainX) 
trainY = np.array(trainY) 


iteration = 100000
learning_rate = 1
lambda_ = 0
dim = trainX.shape[1] + 1 #dim = 163
w = np.zeros(shape = (dim, 1)) # w = (163,1)
Adagrad_sum = np.zeros(shape = (dim,1))
trainX = np.concatenate((np.ones((trainX.shape[0], 1 )), trainX) , axis = 1).astype(float)
trainY = trainY.reshape(trainX.shape[0], 1)
# trainX shape(5652,163) ; trainY shape(5652,1)

for i in range(iteration):
    Y_ = np.dot(trainX,w)
    loss = trainY - Y_
    gradient = (-2)*np.dot(trainX.T,loss) + 2*lambda_*np.sum(w)
    Adagrad_sum += gradient**2
    w -= learning_rate* gradient / (np.sqrt(Adagrad_sum))
    train_loss = np.sqrt( np.sum((loss**2))/trainY.shape[0] )
    print(i,train_loss)

np.save('weight_all_100000.npy',w)     ## save weight