import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sys

# load data
trainX = pd.read_csv(sys.argv[1]).values.astype(float)
trainY = pd.read_csv(sys.argv[2]).values

# scale the age, fnlwgt, captial-gain, hours_per_week
for col in enumerate([0,1,3,5]):
    trainX[:,col[1]] = trainX[:,col[1]]/max(trainX[:,col[1]])

# random forest
#tr_x_train, tr_x_test, tr_y_train,tr_y_test = train_test_split(trainX, trainY, test_size = 0.05, shuffle=True)
tr_x_train = trainX
tr_y_train = trainY
clf = RandomForestClassifier(n_estimators=350, max_depth=27,random_state=0,max_features='log2',oob_score=True)
clf.fit(tr_x_train, tr_y_train)
print("training acc:",clf.score(tr_x_train,tr_y_train))
#print("validation acc:",clf.score(tr_x_test,tr_y_test))

joblib.dump(clf,'best_clf_3.pkl')
