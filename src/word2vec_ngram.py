# -*- coding: utf-8 -*-

import numpy as np
import os 
import pickle
from gensim.models import word2vec  
import logging
import implementations as imp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def main():
    
    DATA_PATH = "../data/"
    NGRAM_RANGE = 2
    N_DIM = 20
    
    train_tweets, labels, test_tweets, nb_tokens = \
            cPickle.load(open(os.path.join(DATA_PATH, 'train_test_{}_gram.pkl'.format(NGRAM_RANGE)), mode='rb'))

    train_data = []    
    for i in range(train_tweets.shape[0]):
        str_tweet = []
        for j in range(train_tweets.shape[1]):
            if train_tweets[i, j] != 0:
               str_tweet.append(str(train_tweets[i, j]))
        
        train_data.append(' '.join(str_tweet))
        
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    model = word2vec.Word2Vec(train_data, size=N_DIM)
    model.save("../data/word2vec_ngram.model")
    
    emb_tweet = np.zeros((len(train_data), N_DIM))
    for i, tweet in enumerate(train_data):
        out_tweet = np.zeros(N_DIM)
        for word in tweet.strip().split():
            try:
                out_tweet += model.wv[word]
            except KeyError:
                pass

        emb_tweet[i, :] = out_tweet
                
    np.random.seed(0)
    ind = np.random.permutation(emb_tweet.shape[0])
    labels = labels[ind]
    emb_tweet = imp.standardize(emb_tweet[ind])
    
    print("Fitting...")
    logistic = LogisticRegression(solver='liblinear')
    logistic.fit(emb_tweet[:199000], labels[:199000])
    
    print("Accuracy = {:.6}".format(np.mean(cross_val_score(logistic, emb_tweet[-10000:], labels[-10000:], cv=5, scoring='accuracy'))))
    
    
if __name__ == '__main__':
    main()





































