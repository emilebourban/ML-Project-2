# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import pickle
from scipy.sparse import *
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd

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

def reduce_dimension(data, embeddings):
    """
    Reduce the dimentionality of the worlds from 21k to 20D
    """
    out = [] 
    for twitt in data:
        tw_vect = np.zeros(embeddings.shape[0])
        # Transforms twitt in a 21k dimentional vector
        # n=0
        # print(twitt)
        tw_vect[twitt] = 1/max(1, len(twitt))
        # for word in twitt:
        #     tw_vect[word] += 1
        #     n += 1
        # tw_vect = tw_vect / max(1, n) 
        # Dimentionality reduction to 20
        out.append(tw_vect @ embeddings)
        
    return np.array(out)


# Loading data
with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)
    
with open(os.path.join(DATA_PATH, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)

# Loading the train data
neg_train = import_data('neg_train.txt', vocab)
pos_train = import_data('pos_train.txt', vocab)

test = import_data('test_data.txt', vocab)      
embed = np.load('embeddings.npy')

# Reducing dimentionality
try:
    pos_train_red = np.load('pos_train_red.npy')
    neg_train_red = np.load('neg_train_red.npy')
    test_red = np.load('test_red.npy')
except: 
    print('fail')
    pos_train_red = reduce_dimension(pos_train, embed)
    neg_train_red = reduce_dimension(neg_train, embed)
    test_red = reduce_dimension(test, embed)
    np.save('pos_train_red', pos_train_red)
    np.save('neg_train_red', neg_train_red)
    # np.save('test_red', neg_train_red)
print('reduction done')
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

logistic = LogisticRegression(solver='liblinear')
logistic.fit(X_train, y_train, )

predictions = logistic.predict(X_test)

accuracy  = sklearn.metrics.accuracy_score(y_test, predictions)

F1 = sklearn.metrics.f1_score(y_test, predictions)
print('Linear; accuracy: {}, F1: {}'.format(accuracy, F1))

# from sklearn import svm
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)

# predictions = clf.predict(X_test) 
# accuracy  = sklearn.metrics.accuracy_score(y_test, predictions)

# F1 = sklearn.metrics.f1_score(y_test, predictions)
# print('accuracy: {}, F1: {}'.format(accuracy, F1))




logistic.fit(X, y)
predictions = logistic.predict(test_red)
predictions = 2 * predictions -1

df = pd.DataFrame(predictions)
df.index.name = 'Id'
df.index +=1
# df.reset_index(inplace=True)


df.columns = ['Prediction']
df.to_csv('predi.csv')