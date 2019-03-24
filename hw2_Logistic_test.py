import pandas as pd
import numpy as np
import csv
import math
import sys

# sigmoid function
def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))

trainX = pd.read_csv(sys.argv[1],engine = 'python').values.astype(float)
testX = pd.read_csv(sys.argv[2],engine = 'python').values.astype(float)#

for col in enumerate([0,1,3,5]):
    testX[:,col[1]] = testX[:,col[1]]/max(trainX[:,col[1]])

w = np.load('best_weight.npy')

# predict testX
testX = np.concatenate((np.ones((testX.shape[0], 1 )), testX) , axis = 1).astype(float)
predict_y = np.apply_along_axis(sigmoid, 1, np.dot(testX,w))
ans = [1 if  i >= 0.5 else 0 for i in predict_y]

# write answer
f = open(sys.argv[3],'w', newline='')
w = csv.writer(f)
title = ['id','label']
w.writerow(title)
for i in range(len(ans)):
    content = ['%d'%(i+1), ans[i]]
    w.writerow(content)
f.close()
