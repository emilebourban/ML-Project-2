# -*- coding: utf-8 -*-

import numpy as np
import os

def import_data(filename, vocab):
    """
    Imports the twitts as lists of dimensions in a list, only imports the words present in vocab
    """
    with open(os.path.join("../data/twitter-datasets/", filename), 'rt') as f:
        out = []
        for line in f:
            temp = []
            for word in line.split():
                try:
                    # Appends the dimention the word is present in
                    temp.append(vocab[word])
                except KeyError:
                    pass
            
            out.append(temp)
    
    return out


def import_text(filename):
    """
    Imports the twitts as lists of worlds in a list, only imports the words present in vocab
    """
    with open(os.path.join("../data/twitter-datasets/", filename), 'rt') as f:
        out = []
        for line in f:
            temp = []
            for word in line.strip().split():
                temp.append(word)
            out.append(temp)
    
    return out


def reduce_dimension(data, embeddings):
    """
    Reduce the dimentionality of the worlds from 21k to 20
    """
    out = [] 
    for twitt in data:
        tw_vect = np.zeros(embeddings.shape[0])
        # Transforms twitt in a 21k dimentional vector
        for word in twitt:
            tw_vect[word] += 1
        # Dimentionality reduction to 20
        out.append(tw_vect @ embeddings)
        
    return np.array(out)


def shuffle_data(X, y, seed=1):
    """
    
    """
    np.random.seed(seed)
    ind = np.random.permutation(y.shape[0])
    X_sh = X[ind]
    y_sh = y[ind]
    return X_sh, y_sh


def split_test_train(X, y, test_fraction=0.8, seed=1):
    """
    
    """
    np.random.seed(seed)
    ind = np.random.permutation(y.shape[0])
    X_tr = X[ind[:int(test_fraction*y.shape[0])]]
    y_tr = y[ind[:int(test_fraction*y.shape[0])]]
    
    X_te = X[ind[int(test_fraction*y.shape[0]):]]
    y_te = y[ind[int(test_fraction*y.shape[0]):]]
    
    return X_tr, X_te, y_tr, y_te


def standardize(X):
    """
    Standardizes the columns of a numpy matrix
    """
    X_st = np.zeros((X.shape[0], X.shape[1]))
    for i  in range(X.shape[1]):
        mean = np.mean(X[:, i])
        std = np.std(X[:, i])
        X_st[:, i] = (X[:, i] - mean) / std
    
    return X_st

def normalize(X):
    """
    Normalizes the columns of a numpy matrix
    """
    X_norm = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        X_norm[:, i] = X[:, i] / np.max(X[:, i])
        
    return X_norm











