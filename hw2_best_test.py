import pandas as pd
import numpy as np
from sklearn.externals import joblib
import csv
import sys

# load data
trainX = pd.read_csv(sys.argv[1],engine='python').values.astype(float)
testX = pd.read_csv(sys.argv[2],engine='python').values.astype(float)

# scale the age, fnlwgt, captial-gain, hours_per_week
for col in enumerate([0,1,3,5]):
    testX[:,col[1]] = testX[:,col[1]]/max(trainX[:,col[1]])

#predict
clf2 = joblib.load('best_clf_3.pkl')

predict_label = clf2.predict(testX)
# write answer
f = open(sys.argv[3],'w', newline='')
w = csv.writer(f)
title = ['id','label']
w.writerow(title)
for i in range(len(predict_label)):
    content = ['%d'%(i+1), predict_label[i]]
    w.writerow(content)
f.close()