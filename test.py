# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import pickle
from scipy.sparse import *

DATA_PATH = "twitter-datasets/"

def import_data(filename, vocab):
    """
    Imports the twitts as lists of worlds in a list
    """
    with open(os.path.join(DATA_PATH, 'neg_train.txt'), 'rt') as f:
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
            tw_vect[word] = 1
        # Dimentionality reduction to 20
        out.append(tw_vect @ embeddings)
        
    return np.array(out)


# Loading data
with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)
    
with open(os.path.join(DATA_PATH, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)

# Loading the train data
pos_train = import_data('pos_train.txt', vocab)
neg_train = import_data('neg_train.txt', vocab)
        
embed = np.load('embeddings.npy')

# Reducing dimentionality
pos_train_red = reduce_dimension(pos_train, embed)
neg_train_red = reduce_dimension(neg_train, embed)
















































