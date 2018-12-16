# -*- coding: utf-8 -*-

import os
import numpy as np

import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM

train_tweets, labels, test_tweets, nb_tokens, emb_matrix = cPickle.load(open("data/train_test_200embedding.pkl", "rb"))

np.random.seed(3)

"""
Same model as before with different seed

"""

model = Sequential()
model.add(Embedding(nb_tokens, 200, input_length=train_tweets.shape[1], weights=[emb_matrix]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_3 = model.predict_proba(train_tweets, batch_size=128)
test_3 = model.predict_proba(test_tweets)

"""
Dump the results of model 3

"""

cPickle.dump(train_3, open('results/train_conv3_pretrained.dat', 'wb'))
cPickle.dump(test_3, open('results/test_conv3_pretrained.dat', 'wb'))



