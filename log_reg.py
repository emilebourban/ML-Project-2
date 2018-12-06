# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from src import implementations as imp

DATA_PATH = "data/"

with open(os.path.join(DATA_PATH, "twitts_train.pkl"), 'rb') as f:
    data = pickle.load(f)
    twitt_data = data['twitts']
    smileys = data['smileys']
    
logistic =  LogisticRegression(solver='lbfgs')
logistic.fit(twitt_data, smileys)

sm_pred = logistic.predict(twitt_data)
display("Accuracy = {}".format(1 - np.fabs(sm_pred - smileys).sum() * 0.5 / twitt_data.shape[0]) )