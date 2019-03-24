import pandas as pd
import numpy as np
import math
import csv
import sys

# load data
trainX = pd.read_csv(sys.argv[1],engine='python').values.astype(float)
trainY = pd.read_csv(sys.argv[2],engine='python').values
testX = pd.read_csv(sys.argv[3],engine='python').values.astype(float)

# training data preprocessing
# scale the age, fnlwgt, captial-gain, hours_per_week
for col in enumerate([0,1,3,5]):
    trainX[:,col[1]] = trainX[:,col[1]]/max(trainX[:,col[1]])

for col in enumerate([0,1,3,5]):
    testX[:,col[1]] = testX[:,col[1]]/max(trainX[:,col[1]])

# seperate data into class_0 and class_1 
class_0 = []
class_1 = []
for i in range(len(trainY)):
    if trainY[i] == 0:
        class_0.append(trainX[i])
    elif trainY[i] == 1:
        class_1.append(trainX[i])

class_0 = np.array(class_0) # shape = (24720, 106)
class_1 = np.array(class_1) # shape = (7841, 106)

# get mean & var
mean_0 = np.mean(class_0, axis=0)
mean_1 = np.mean(class_1, axis=0)

n = class_0.shape[1]
cov_0 = np.zeros((n,n))
cov_1 = np.zeros((n,n))

for i in range(class_0.shape[0]):
    cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]
for i in range(class_1.shape[1]):
    cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), ([class_1[i] - mean_1])) / class_1.shape[1]

cov = (class_0.shape[0]*cov_0 + class_1.shape[0]*cov_1)/(class_0.shape[0]+class_1.shape[0])

# get w and b
w = np.dot((mean_0 - mean_1), np.linalg.inv(cov))
b = (-0.5)*mean_0.dot(np.linalg.inv(cov)).dot(np.transpose(mean_0))+\
    0.5*mean_1.dot(np.linalg.inv(cov)).dot(np.transpose(mean_1))+\
    np.log(float(class_0.shape[0]/class_1.shape[0]))

# sigmoid function
def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))

# predict ans
z = (np.dot(testX, w) + b).reshape(testX.shape[0],1)
#predicted = 1/(1 + np.exp(-1*z))
predicted = np.apply_along_axis(sigmoid, 1, z)
ans = [1 if  i >= 0.5 else 0 for i in predicted]

# write answer
f = open(sys.argv[4],'w', newline='')
w = csv.writer(f)
title = ['id','label']
w.writerow(title)
for i in range(len(ans)):
    content = ['%d'%(i+1), ans[i]]
    w.writerow(content)
f.close()

