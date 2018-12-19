# -*- coding: utf-8 -*-

import os
import numpy as np
import _pickle as cPickle
import xgboost as xgb
from src import submisson as sub

def main():
    train_files = [file for file in os.listdir("results/train/") if 'ep' in file and '6' not in file]
    test_files = [file for file in os.listdir("results/test/") if 'ep' in file and '6' not in file]
    
    print("Training on {}".format(train_files))
    train = []
    for file in train_files:
        train.append(cPickle.load(open("results/train/{}".format(file), "rb")))
    
    print("Predicting on {}".format(test_files))
    test = []
    for file in test_files:
        test.append(cPickle.load(open("results/test/{}".format(file), "rb")))
        
    train = np.hstack(train)
    test = np.hstack(test)
    
    y = np.array(train.shape[0]//2 * [0] + train.shape[0]//2 * [1])
    
    model = xgb.XGBClassifier().fit(train, y)

    y_pred = model.predict(test)
    y_pred = 2*y_pred - 1
    sub.create_submission(np.arange(1,test.shape[0]+1),y_pred, 'final_pred')    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    