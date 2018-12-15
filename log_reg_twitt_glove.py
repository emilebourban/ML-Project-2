# -*- coding: utf-8 -*-

import os
import numpy as np
import _pickle as cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src import implementations as imp



def main():
    """
    This is just a quick test to check whether the preprocessing, tokenizing and embeding worked
    """
    DATA_PATH = "data/"
    EMB_DIM = 200    
    
    print("Loading pickled data, embedings: {}...".format(EMB_DIM))
    train_tweets, labels, test_tweets, nb_tokens, emb_matrix = \
        cPickle.load(open(os.path.join(DATA_PATH, "train_test_{}embedding.pkl".format(EMB_DIM)), mode='rb'))
        
    print("Embedding...")      
    train_data = np.zeros((train_tweets.shape[0], EMB_DIM))    
    for i, tweet in enumerate(train_tweets):
        
        temp_tweet = np.zeros(EMB_DIM)
        for token in tweet:            
            if token != 0:
                temp_tweet = temp_tweet + emb_matrix[token-1, :]
                
        train_data[i, :] = temp_tweet
        
    np.random.seed(0)
    ind = np.random.permutation(train_tweets.shape[0])
    labels = labels[ind]
    train_data = imp.standardize(train_data[ind])
    
    print("Logistic regresion fitting...")
    logistic = LogisticRegression(solver='liblinear')
    logistic.fit(train_data[:200000], labels[:200000])
    
    print("Accuracy is: {}".format(np.mean(cross_val_score(logistic, train_data[-10000:], labels[-10000:], cv=5, scoring='accuracy'))))

if __name__ == '__main__':    
    main()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    