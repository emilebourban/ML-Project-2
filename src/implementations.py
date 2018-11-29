# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle

def import_data(filename, vocab):
    """
    Imports the twitts as lists of worlds in a list
    """
    with open(os.path.join("twitter-datasets/", filename), 'rt') as f:
        out = []
        for line in f:
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

















