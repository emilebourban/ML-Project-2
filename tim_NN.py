# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import pickle
from scipy.sparse import *
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

DATA_PATH = "twitter-datasets/"

def import_data(filename, vocab):
    """
    Imports the twitts as lists of worlds in a list
    """
    with open(os.path.join(DATA_PATH, filename), 'rt', encoding="utf8") as f:
        out = []
        for line in f:
            # print(line)
            temp = []
            for word in line.split():
                try:
                    temp.append(vocab[word])
                except KeyError:
                    pass
            
            out.append(temp)
    
    return out


def reduce_dimension2(data, embeddings, max_dim = 128):
    """
    Reduce the dimentionality of the worlds from 21k to 20D
    """
    out = [] 
    j=0
    n= len(data)
    for twitt in data:
        if (j/n*100)==int(j/n*100):
            print(str(j/n*100) + 'percent done')
        j +=1
        tw_vect = np.zeros([ max_dim, embeddings.shape[0]])
        # Transforms twitt in a 21k dimentional vector
        # n=0
        # print(twitt)
        # tw_vect[twitt] = 1/max(1, len(twitt))
        i = 0
        for word in twitt:
            tw_vect[i, word] = 1
            i += 1
        # tw_vect = tw_vect / max(1, n) 
        # Dimentionality reduction to 20
        out.append(tw_vect @ embeddings)
        
    return np.array(out)


# Loading data
with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)
    
with open(os.path.join(DATA_PATH, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)


max_dim =63
# Loading the train data
do_embeding = False
if do_embeding:
    neg_train = import_data('neg_train.txt', vocab)
    pos_train = import_data('pos_train.txt', vocab)

    test = import_data('test_data.txt', vocab)      
    embed = np.load('embeddings.npy')

    pos_train_red = reduce_dimension2(pos_train, embed, max_dim)
    print('Pos reduced')
    neg_train_red = reduce_dimension2(neg_train, embed, max_dim)
    print('Neg reduced')
    test_red = reduce_dimension2(test, embed, max_dim)
    print('Test reduced')
    np.save('pos_train_red_keep_pos', pos_train_red)
    np.save('neg_train_red_keep_pos', neg_train_red)
    np.save('test_red_keep_pos', neg_train_red)

# Reducing dimentionality

pos_train_red = np.load('pos_train_red_keep_pos.npy')
neg_train_red = np.load('neg_train_red_keep_pos.npy')
test_red = np.load('test_red_keep_pos.npy')

word_vect_dim = test_red.shape[-1]
output_size =2
continue_ = True
if continue_:
    print('Continuing')
    train_red = np.concatenate((pos_train_red, neg_train_red))
    # len(train_red)
    y=np.zeros(len(train_red))
    y[:int(len(train_red)/2)]=1
    y = y.reshape(-1,1)


    s = np.arange(y.shape[0])
    np.random.shuffle(s)

    X = train_red[s]
    y = y[s]
    p=0.8

    n=int(len(y)*p)
    X_train = X[:n,:]
    y_train = y[:n,:]
    X_test = X[n:,:]
    y_test = y[n:,:]

    Y_train = np_utils.to_categorical(y_train, output_size)
    Y_test = np_utils.to_categorical(y_test, output_size)

    X_train = X_train.reshape(X_train.shape[0], max_dim, word_vect_dim, 1)
    X_test = X_test.reshape(X_test.shape[0], max_dim, word_vect_dim, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #model
    model = Sequential()

    nb_filter = 32
    x_size = 4
    y_size = word_vect_dim
    model.add(Convolution2D(nb_filter, (x_size, y_size), activation='relu', input_shape=(max_dim, word_vect_dim,1)))
    print( model.output_shape)
    # model.add(Convolution2D(32, 3, 28, activation='relu'))
    i=0
    print('test{}'.format(i))
    i +=1
    model.add(MaxPooling2D(pool_size=(26,1)))
    print( model.output_shape)
    # model.add(Dropout(0.25))
    print('test{}'.format(i)) # 1
    i +=1
    model.add(Flatten())
    print( model.output_shape) # 2

    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    print('test{}'.format(i)) # 3
    i +=1
    model.compile(loss= 'binary_crossentropy',# 'categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    print('test{}'.format(i)) # 4
    i +=1
    # Train model on train data
    model.fit(X_train, Y_train, 
            batch_size=64, nb_epoch=10, verbose=1)

    # Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)

    print(score)

test_red_2 = test_red.reshape(test_red.shape[0], max_dim, word_vect_dim, 1)
predictions = model.predict(test_red_2).argmax(axis=1)
predictions = 2 * predictions -1
df = pd.DataFrame(predictions)
df.index.name = 'Id'
df.index +=1
# df.reset_index(inplace=True)


df.columns = ['Prediction']
df.to_csv('predi.csv')
    