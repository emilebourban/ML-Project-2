# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
from scipy.sparse import *
from src import implementations as imp

DATA_PATH = "data/"

# Loading data
with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)
    
with open(os.path.join(DATA_PATH, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)

# Loading the train data
pos_train = imp.import_data('pos_train.txt', vocab)
neg_train = imp.import_data('neg_train.txt', vocab)
        
embed = np.load(os.path.join(DATA_PATH, 'embeddings.npy'))

# Reducing dimentionality
pos_train_red = imp.reduce_dimension(pos_train, embed)
neg_train_red = imp.reduce_dimension(neg_train, embed)

# Shuffling pos and neg data
twitt_all = np.concatenate((pos_train_red, neg_train_red), axis=0)
smiley_all = np.concatenate((np.ones(pos_train_red.shape[0]), 
                             -np.ones(neg_train_red.shape[0])), axis=0)

twitt_data, smileys = imp.shuffle_data(twitt_all, smiley_all)

# Saves a file with all the twitts and coresponding smileys
pickle.dump({'twitts':twitt_data, 'smileys':smileys}, 
            open(os.path.join(DATA_PATH, "twitts_train.pkl"), "wb") )














































