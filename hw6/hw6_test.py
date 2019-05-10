# -*- coding: utf-8 -*
import sys
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Embedding, Input,InputLayer
from keras.layers import Conv1D, MaxPooling1D, Embedding, LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,History ,ModelCheckpoint
from keras.models import load_model
import pandas as pd
import jieba

MAX_SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 300
#bash hw4_test.sh <test_x file> <dict.txt.big file> <output file>

test_data_path = sys.argv[1]  #sys.argv[1] 'test_x.csv'
dict_path = sys.argv[2]       #sys.argv[2] 'dict.txt.big'
prediction_path = sys.argv[3] # sys.argv[3] 'test_4_output.csv'


texts = list()
labels = list()
fileTrainSeg=[]
jieba.load_userdict(dict_path)


with open(test_data_path,encoding='utf8') as f:
    next(f)
    for line in f:
        item = line.strip().split(',')
        texts.append(item[1])
print(len(texts))
print('load data success!')
filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n。 \\//，、嗎阿ㄇㄅ'

#fixdata
for i in range(len(texts)):
    for c in filters:
        texts[i] = texts[i].replace(c, '')

fileTrainSegs=[]
for i in range(len(texts)):
    fileTrainSegs.append((list(jieba.cut(texts[i],cut_all=False))))

print('preprocess data success!')

word2vec = Word2Vec.load('model_300.bin')

dataVector = np.zeros((len(fileTrainSegs), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for n in range(len(fileTrainSegs)):
    for i in range(min(len(fileTrainSegs[n]), MAX_SEQUENCE_LENGTH)):
        try:
            vector = word2vec[fileTrainSegs[n][i]]
            dataVector[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
        except KeyError as e:
            pass
            #print ('Word', texts[n][i], 'is not in dictionary.')
    if n % 10000==0:
        print(n)

np.save('dataVector.npy',dataVector)

model4 = load_model('last_10_300_80_model-00007-0.76325.h5')
model5 = load_model('10_300_80_model-00005-0.76125.h5')
model6 = load_model('10_300_80_model-00007-0.76708.h5')
model7 = load_model('300_80_model-00007-0.76392.h5')

p4 = model4.predict(dataVector,verbose=1)
print(p4[:10])
p5 = model5.predict(dataVector,verbose=1)
print(p5[:10])
p6 = model6.predict(dataVector,verbose=1)
print(p6[:10])
p7 = model7.predict(dataVector,verbose=1)
print(p7[:10])

pred_y_prob = (p4+p5+p6+p7)/4
pred_y = np.argmax(pred_y_prob,axis=1)
submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_y
submission.to_csv(prediction_path,index=False)

print('test finish!')
#reference 
# https://github.com/thtang/ML2017FALL/
# https://hackmd.io/s/HyrW0uyeN