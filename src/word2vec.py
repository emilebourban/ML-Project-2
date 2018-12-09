# -*- coding: utf-8 -*-

import numpy as np
import os 
import pickle
from gensim.models import word2vec  
import logging
import implementations as imp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
    
def main():
    DATA_PATH = "../data"
    N_DIM = 100
    N_TWITT = 200000
    
    with open(os.path.join(DATA_PATH, "vocab.pkl"), 'rb') as f:
            vocab = pickle.load(f)
            vocab_size = len(vocab)
    
    train_twitts = imp.import_text('train_pos.txt', vocab)
    train_twitts.extend(imp.import_text('train_neg.txt', vocab))
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    model = word2vec.Word2Vec(train_twitts, size=N_DIM)
    
    twitt_data = np.zeros((N_TWITT, N_DIM))
    train_files = ['train_pos.txt', 'train_neg.txt']
    for i, file in enumerate(train_files):
        
        with open(os.path.join(DATA_PATH, "twitter-datasets", file), 'rt') as f:
            for l, line in enumerate(f):
                twitt = np.zeros(N_DIM)
                for word in line.strip().split():
                    try:
                        twitt += model.wv[word]
                    except:
                        continue
                twitt_data[i*100000 + l , :] = twitt
                
    smileys = np.concatenate((np.ones(N_TWITT//2), -np.ones(N_TWITT//2)), axis=None)
    
    twitt_tr, twitt_te, smileys_tr, smileys_te = split_test_train(twitt_data, smileys)
    
    logistic = LogisticRegression(solver='liblinear')
    logistic.fit(twitt_tr, smileys_tr)
    
    print("Accuracy = {:.6}".format(np.mean(cross_val_score(logistic, twitt_te, smileys_te, cv=5, scoring='accuracy'))))
            
if __name__ == '__main__':
    main()
            
            
            
            
            
            
            
            
            
            
            
            