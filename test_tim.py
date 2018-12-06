# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import pickle
from scipy.sparse import *
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
from src import implementations as imp

DATA_PATH = "data/"
RESULT_PATH = "results/"



# Loading data
with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)
    
with open(os.path.join(DATA_PATH, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)

# Loading the train data
neg_train = imp.import_data('train_neg.txt', vocab)
pos_train = imp.import_data('train_pos.txt', vocab)

test = imp.import_data('test_data.txt', vocab)      
embed = np.load(os.path.join(DATA_PATH,'embeddings.npy'))

# Reducing dimentionality
try:
    pos_train_red = np.load(os.path.join(DATA_PATH,'pos_train_red.npy'))
    neg_train_red = np.load(os.path.join(DATA_PATH,'neg_train_red.npy'))
    test_red = np.load('test_red.npy')
except: 
    print('fail')
    pos_train_red = imp.reduce_dimension(pos_train, embed)
    neg_train_red = imp.reduce_dimension(neg_train, embed)
    test_red = imp.reduce_dimension(test, embed)
    np.save(os.path.join(DATA_PATH,'pos_train_red'), pos_train_red)
    np.save(os.path.join(DATA_PATH,'neg_train_red'), neg_train_red)
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
df.to_csv(os.path.join(RESULT_PATH,'predi.csv'))