# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os 
import pickle
from gensim.models import word2vec  
import logging
import implementations as imp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sklearn
from importlib import reload
reload(imp)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model


def split_test_train(X, y, train_fraction=0.8, seed=1):
    """
    
    """
    np.random.seed(seed)
    ind = np.random.permutation(y.shape[0])
    try:
        X_tr = X[ind[:int(train_fraction*y.shape[0])]]
        y_tr = y[ind[:int(train_fraction*y.shape[0])]]
        
        X_te = X[ind[int(train_fraction*y.shape[0]):]]
        y_te = y[ind[int(train_fraction*y.shape[0]):]]
    except:
        X_tr = X[ind[:int(train_fraction*y.shape[0]/10)]]
        y_tr = y[ind[:int(train_fraction*y.shape[0]/10)]]
        
        X_te = X[ind[int(train_fraction*y.shape[0]/10):int(y.shape[0]/10)]]
        y_te = y[ind[int(train_fraction*y.shape[0]/10):int(y.shape[0]/10)]]        
    
    return X_tr, X_te, y_tr, y_te
    
# def main():
DATA_PATH = "../data"
word_vect_dim = 100
# N_TWITT = 200000

with open(os.path.join(DATA_PATH, "vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)
full_ = False
train_twitts = imp.import_text('cl_train_pos'+('_full' if full_ else '')+'.txt')
N_POS=len(train_twitts)
train_twitts.extend(imp.import_text('cl_train_neg'+('_full' if full_ else '')+'.txt'))
N_NEG =len(train_twitts)-N_POS
train_twitts.extend(imp.import_text('cl_test_data.txt'))
N_test = len(train_twitts)-N_POS-N_NEG
N_TWITT = len(train_twitts)
# Creates trains and saves the model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
load_ = True
if load_:
	model = word2vec.Word2Vec.load("../data/word2vec{}.model".format(word_vect_dim))
else:
	model = word2vec.Word2Vec(train_twitts, size=word_vect_dim, iter=20)
	model.save("../data/word2vec{}.model".format(word_vect_dim))

N_TWITT = N_TWITT-10000
sentence_dim = 48
twitt_data = np.zeros((N_POS + N_NEG, sentence_dim, word_vect_dim))
train_files = ['cl_train_pos'+('_full' if full_ else '')+'.txt', 'cl_train_neg'+('_full' if full_ else '')+'.txt']

for i, file in enumerate(train_files):
    
    with open(os.path.join(DATA_PATH, "twitter-datasets", file), 'rt', encoding="utf8") as f:

        for l, line in enumerate(f):
            twitt = np.zeros([sentence_dim, word_vect_dim])
            # n= len(line.strip().split())
            # print(n)
            j=0
            for word in line.strip().split():
                try:
                    twitt[j , :] = model.wv[word] #/ n
                    j += 1
                except:
                    continue
            # twitt
            twitt_data[i*N_POS + l, : , :] = twitt
print('loading is over')          
smileys = np.concatenate((np.ones(N_TWITT//2), -np.ones(N_TWITT//2)), axis=None)
smileys[smileys<0]=0
twitt_tr, twitt_te, smileys_tr, smileys_te = split_test_train(twitt_data, smileys)


output_size =2

Y_train = np_utils.to_categorical(smileys_tr, output_size)
Y_test = np_utils.to_categorical(smileys_te, output_size)

X_train = twitt_tr.reshape(twitt_tr.shape[0], sentence_dim, word_vect_dim, 1)
X_test = twitt_te.reshape(twitt_te.shape[0], sentence_dim, word_vect_dim, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('Creat model')
#model
model_NN = Sequential()

nb_filter = 32
x_size = 5
y_size = word_vect_dim
model_NN.add(Convolution2D(nb_filter, (x_size, y_size), activation='relu', input_shape=(sentence_dim, word_vect_dim,1)))
print( model_NN.output_shape)
# model_NN.add(Convolution2D(32, 3, 28, activation='relu'))
i=0
print('test{}'.format(i))
i +=1
model_NN.add(MaxPooling2D(pool_size=(44,1)))
print( model_NN.output_shape)
model_NN.add(Dropout(0.25))
print('test{}'.format(i)) # 1
i +=1
model_NN.add(Flatten())
print( model_NN.output_shape) # 2

model_NN.add(Dense(128, activation='relu'))
model_NN.add(Dropout(0.5))
model_NN.add(Dense(output_size, activation='softmax'))

print('test{}'.format(i)) # 3
i +=1
model_NN.compile(loss= 'binary_crossentropy',# 'categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
print('test{}'.format(i)) # 4
i +=1
# Train model on train data
model_NN.fit(X_train, Y_train, 
        batch_size=64, nb_epoch=10, verbose=1)

# Evaluate model on test data
score = model_NN.evaluate(X_test, Y_test, verbose=0)

print(score)

# model_NN.save('my_model.h5')
# model_NN2 = load_model('my_model.h5')


########################################################
test_file = 'cl_test_data.txt'
# for i, file in enumerate(test_files):
twitt_data_test = np.zeros((10000, sentence_dim, word_vect_dim))  

test_file = 'cl_test_data.txt'
with open(os.path.join(DATA_PATH, "twitter-datasets", test_file), 'rt', encoding="utf8") as f:

        for l, line in enumerate(f):
            twitt = np.zeros([sentence_dim, word_vect_dim])
            n= len(line.strip().split())
            # print(n)
            j=0
            for word in line.strip().split():
                try:
                    # print(n)
                    twitt[j , :] = model.wv[word] #/ n
                    j += 1
                except:
                    continue
            # twitt
            twitt_data_test[l, : , :] = twitt



test_red_2 = twitt_data_test.reshape(twitt_data_test.shape[0], sentence_dim, word_vect_dim, 1)
predictions = model_NN.predict(test_red_2).argmax(axis=1)

# logistic = LogisticRegression(solver='liblinear')#, C=1/4)
# logistic.fit(twitt_tr, smileys_tr)
# predictions = logistic.predict(twitt_te)
# accuracy  = sklearn.metrics.accuracy_score(smileys_te, predictions)
# print('Linear; accuracy: {}'.format(accuracy))
# predictions = 2 * predictions -1

# print('Mean prediction {}, close to zero is good sign'.format(predictions.mean()))


# test_file = 'cl_test_data.txt'
# # for i, file in enumerate(test_files):
# twitt_data_test = np.zeros((10000, word_vect_dim*sentence_dim))    
# with open(os.path.join(DATA_PATH, "twitter-datasets", test_file), 'rt', encoding="utf8") as f:

#     for l, line in enumerate(f):
#         twitt = np.zeros([word_vect_dim,sentence_dim])
#         # n= len(line.strip().split())
#         # print(n)
#         j = 0
#         for word in line.strip().split():
#             try:
#                 twitt[:,j%sentence_dim] += model.wv[word] #/ n
#                 j += 1
#             except:
#                 continue
#         twitt_data_test[l , :] = twitt.reshape(word_vect_dim * sentence_dim)

# twitt_tr2, twitt_te2, smileys_tr2, smileys_te2 = split_test_train(twitt_data, smileys,	train_fraction=1)
# logistic = LogisticRegression(solver='liblinear')#, C=1/4)
# logistic.fit(twitt_tr2, smileys_tr2)
# predictions = logistic.predict(twitt_data_test)
predictions = 2 * predictions -1

print('Mean prediction {}, close to zero is good sign'.format(predictions.mean()))
df = pd.DataFrame(predictions)
df.index.name = 'Id'
df.index +=1
# df.reset_index(inplace=True)


df.columns = ['Prediction']
df.to_csv('predi.csv')



    # print("Accuracy = {:.6}".format(np.mean(cross_val_score(logistic, twitt_te, smileys_te, cv=5, scoring='accuracy'))))
            
# if __name__ == '__main__':
#     main()
            
            
            
            
            
            
            
            
            
            
            
            