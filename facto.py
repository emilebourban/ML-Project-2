import numpy as np
import os
import pickle
import random
DATA_PATH = 'twitter-datasets'
from sklearn.decomposition import NMF

with open(os.path.join(DATA_PATH, 'cooc.pkl'), 'rb') as f:
    cooc = pickle.load(f)

model = NMF(n_components=50, init='random', random_state=0, l1_ratio=0.1)
W = model.fit_transform(cooc)
H = model.components_
np.save('embeddings', W)