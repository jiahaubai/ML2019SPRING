import pandas as pd
import numpy as np
import csv
import sys

input_path = sys.argv[1]
ouput_path = sys.argv[2]

w = np.load('weight_all_100000.npy')
df = pd.read_csv(input_path,header=None,engine = 'python').iloc[:,2:].values
#print(df)

testX = [[] for i in range(240)]# idx:0-239
#print(data)
count = 0

for row in range(len(df)):
    
    if row%18 == 0 and row > 0:
        count+=1
    
    temp = list(df[row])
    for i in range(len(temp)):
        if temp[i] == 'NR':
            temp[i] = 0
        if i>=0:                                  #<--- i>=4
            temp[i] = float(temp[i])
            testX[ count ].append(temp[i])

testX = np.array(testX)
testX = np.concatenate((np.ones((240,1)), testX), axis=1).astype(float)
ans = np.dot(testX,w)

with open(ouput_path, 'w', newline = '') as fp:
    fieldnames = ['id','value']
    writer = csv.DictWriter(fp, fieldnames = fieldnames)
    writer.writeheader()
    
    dict = {'id':'', 'value':''}
    for n in range(len(testX)):
        
        dict['id'] = 'id_%d'%(n)
        dict['value'] = ans[n][0]
        print(n,ans[n][0])
        
        writer.writerow(dict)
        dict.clear() 