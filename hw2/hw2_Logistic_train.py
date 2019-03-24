import pandas as pd
import numpy as np
import math
import sys

# load data
trainX = pd.read_csv(sys.argv[1]).values.astype(float)
trainY = pd.read_csv(sys.argv[2]).values

# training data preprocessing
# scale the age, fnlwgt, captial-gain, hours_per_week
for col in enumerate([0,1,3,5]):
    trainX[:,col[1]] = trainX[:,col[1]]/max(trainX[:,col[1]])
    
# sigmoid function
def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))

# decare some parameters of gradient descent 
iteration = 1200
learning_rate = 1
lambda_ = 0
dim = trainX.shape[1] + 1 #dim = 106+1
w = np.zeros(shape = (dim, 1)) # w = (107,1)
Adagrad_sum = np.zeros(shape = (dim,1)) # Adagrad_sum = (107,1)
trainX = np.concatenate((np.ones((trainX.shape[0], 1 )), trainX) , axis = 1).astype(float)
# trainX shape(32561, 107) ; trainY shape(32561, 1)

best_acc = 0
best_iter = 0
# iteration
for i in range(iteration):
    
    if i >= 1000: learning_rate = 0.01
    z = np.dot(trainX,w)
    Y_ = np.apply_along_axis(sigmoid, 1, z).reshape(trainY.shape[0],1)
    loss = trainY - Y_
    gradient = (-2)*np.dot(trainX.T,loss) + 2*lambda_*np.sum(w)
    Adagrad_sum += gradient**2
    w -= learning_rate* gradient / (np.sqrt(Adagrad_sum))
    train_loss = np.sqrt( np.sum((loss**2))/trainY.shape[0] )
    if i%10==0:
        ans = np.array([1 if  i >= 0.5 else 0 for i in Y_ ])
        ans = ans.reshape(32561, 1)
        #acc = trainY-ans
        acc = (list(trainY-ans).count(0))/len(ans)
        print(i,train_loss,acc)
        if acc > best_acc:
            np.save('best_weight.npy',w)
            best_iter = i
            best_acc = acc

print("best_iter:",best_iter)
print("best_acc:",best_acc)
