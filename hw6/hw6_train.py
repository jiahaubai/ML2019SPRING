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
from keras.callbacks import EarlyStopping,History ,ModelCheckpoint,CSVLogger
from keras.utils import plot_model
import jieba

MAX_SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 300
MIN_COUNT = 3
VALIDATION_SPLIT = 0.1
#bash hw4_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>

train_path = sys.argv[1] #'train_x.csv'
label_path = sys.argv[2] #'train_y.csv'
test_path = sys.argv[3] #'test_x.csv'
dict_path = sys.argv[4] #'dict.txt.big'

texts = list()
labels = list()

fileTrainSeg=[]
with open(train_path,encoding = 'utf8') as f:
    next(f)
    for line in f:
        item = line.strip().split(',')
        texts.append(item[1])

with open(test_path,encoding = 'utf8') as f:
    # Read line by line.
    next(f)
    for line in f:
        item = line.strip().split(',')
        texts.append(item[1])
print()

with open(label_path) as f:
    # Read line by line.
    next(f)
    for line in f:
        item = line.strip().split(',')
        labels.append(item[1])
print('data read finish',len(texts),len(labels))

jieba.load_userdict(dict_path)
#jieba.enable_parallel(8)

print('start filter preprocess')
filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n。 \\//，、嗎阿ㄇㄅ'
#fixdata
for i in range(len(texts)):
    for c in filters:
        texts[i] = texts[i].replace(c, '')

fileTrainSegs=[]
for i in range(len(texts)):
    fileTrainSegs.append((list(jieba.cut(texts[i],cut_all=False))))

print('finish filter preprocess')


word2vec = Word2Vec.load('model_300.bin')
print(word2vec)

print('word->vector')

dataVector = np.zeros((120000, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for n in range(120000):
    for i in range(min(len(fileTrainSegs[n]), MAX_SEQUENCE_LENGTH)):
        try:
            vector = word2vec[fileTrainSegs[n][i]]
            dataVector[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
        except KeyError as e:
            pass
            #print ('Word', texts[n][i], 'is not in dictionary.')
    if n % 10000==0:
        print(n)


labels = to_categorical(np.asarray(labels))
numClasses = labels.shape[1]

for i in range(5):
    print('train round:',str(i))
    model = Sequential()

    model.add(GRU(units=128, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(units=256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numClasses, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    check_save  = ModelCheckpoint("300_80_model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
    early_stop = EarlyStopping(monitor="val_acc", patience=4,mode='max')
    #csv_logger = CSVLogger('trainhw4')
    fitHistory = model.fit(dataVector, labels,
        batch_size=128, 
        epochs=16,verbose=2,
        validation_split=0.1,shuffle=True,
        callbacks=[check_save,early_stop]
        )
    
    