import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sys
import csv

# load data
trainX = pd.read_csv('X_train').values.astype(float)#sys.argv[1],engine='python'
trainY = pd.read_csv('Y_train').values#sys.argv[2],engine='python'
testX = pd.read_csv('X_test').values.astype(float)#sys.argv[3],engine='python'

# scale the age, fnlwgt, captial-gain, hours_per_week
for col in enumerate([0,1,3,5]):
    trainX[:,col[1]] = trainX[:,col[1]]/max(trainX[:,col[1]])
for col in enumerate([0,1,3,5]):
    testX[:,col[1]] = testX[:,col[1]]/max(trainX[:,col[1]])

# random forest
#tr_x_train, tr_x_test, tr_y_train,tr_y_test = train_test_split(trainX, trainY, test_size = 0.05, shuffle=True)
tr_x_train = trainX
tr_y_train = trainY
clf = RandomForestClassifier(n_estimators=350, max_depth=27,random_state=0,max_features='log2',oob_score=True)
clf.fit(tr_x_train, tr_y_train)
print("training acc:",clf.score(tr_x_train,tr_y_train))
#print("validation acc:",clf.score(tr_x_test,tr_y_test))

joblib.dump(clf,'best.joblib')
'''
clf2 = load('best.joblib')  
predict_label = clf2.predict(testX)
# write answer
f = open('ans.csv','w', newline='')#sys.argv[4]
w = csv.writer(f)
title = ['id','label']
w.writerow(title)
for i in range(len(predict_label)):
    content = ['%d'%(i+1), predict_label[i]]
    w.writerow(content)
f.close()
'''
