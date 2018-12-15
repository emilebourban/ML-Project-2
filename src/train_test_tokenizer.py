import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import _pickle as cPickle
from nltk import ngrams
from collections import Counter


def import_tweets(filepath, FULL=False):
    """
    
    """
    if FULL:
        train_files = [file for file in os.listdir(filepath) if 'full' in file]
        train_size = 2500000
    else:
        train_files = [file for file in os.listdir(filepath) if 'full' not in file]        
        train_size = 200000
        
    test_file = [file for file in os.listdir(filepath) if 'test' in file]
    
    train_tweets = []; test_tweets = []
    
    labels = np.array(train_size // 2 * [0] + train_size // 2 * [1])
    
    
    print("Loading tweet data...")
    for file in train_files:
        with open(os.path.join(filepath, file), mode='rt', encoding='utf-8') as f:
            for line in f:
                train_tweets.append(line.strip())
                
    with open(os.path.join(filepath, test_file[0]), mode='rt', encoding='utf-8') as f:
        for line in f:
            test_tweets.append(line.strip())
            
    return train_tweets, test_tweets, labels
    
    
def tokenize(train_tweets, test_tweets, max_word=None):
    """
    
    """
    if max_word == None:
        tokenizer = Tokenizer(filters='')
    else:
        tokenizer = Tokenizer(nb_words=max_word, filters='')
    
    print("Fitting tokenizer...")
    tokenizer.fit_on_texts(train_tweets)
    word_index = tokenizer.word_index
    nb_token = len(word_index)
    print("Found {} unique tokens.".format(nb_token))

    train_tweets_tokenized = tokenizer.texts_to_sequences(train_tweets)
    test_tweets_tokenized = tokenizer.texts_to_sequences(test_tweets)
    
    return train_tweets_tokenized, test_tweets_tokenized, word_index


def create_ngram_dic(tweets, max_token, ngram_range=2, min_occ=0, max_occ=None, n_first=None):
    '''
    Creates a dictionary that has ngram of tokens for keys and an unique int for value
    IN: 
        tweets: list of list of tokenized tweets
        max_token: int, max value of tokens in tweets
        ngram_range: which ngrams to compute
        min_occ: when not None, sets the minium of occurence of ngrams to keep
        max_occ: when not None, sets the maximum of occurence of ngrams to keep
        n_first: when not None, sets the number of most occuring ngrams to keep
    OUT: 
        ngram_dic: dictionary that has ngram of tokens for keys and an unique int for value
    '''
    ngram_list =[]
    for tweet in tweets:
        for i in range(2, ngram_range + 1):
            ngram_list.append(ngrams(tweet, i))
    
    counter = Counter(ngram_list)
    
    new_ngram_list = []
    if n_first:
        new_ngram_list = [val[0] for val in counter.most_common(n_first)]
    elif max_occ:
        for key, val in counter:
            if val >= min_occ and val <= max_occ:

                new_ngram_list.append(key)
               
    new_tokens = range(max_token+1, len(new_ngram_list)+max_token+1)
    ngram_dic = dict(zip(new_ngram_list, new_tokens))
    
    return ngram_dic
    
        
def add_ngrams(train_tweets, test_tweets, max_token, ngram_range=2, min_occ=0, max_occ=None, n_first=None):
    '''
    appends a token of the ngrams in the tweet. Tokens that are appended can be
    restricted in function of number of occurence.
    IN: see create_ngram_dic
    OUT: a list of list of tokens for each tweet with ngram tokens included
    '''
    ngram_dic = create_ngram_dic(train_tweets, max_token, ngram_range=ngram_range, min_occ=min_occ, max_occ=max_occ, n_first=n_first)
    
    new_train_tweets = []
    new_test_tweets = []
    
    for tweet in train_tweets:
        ngram_list = []
        new_tweet = tweet
        for i in range(2, ngram_range + 1):
            ngram_list.append(ngrams(tweet, i))
            
        ngram_set = set(ngram_list)
        
        for ngram in ngram_set:
            try:
                new_tweet.append(ngram_dic[ngram])
            except KeyError:
                continue
        new_train_tweets.append(new_tweet)
        
    for tweet in test_tweets:
        ngram_list = []
        new_tweet = tweet
        for i in range(2, ngram_range + 1):
            ngram_list.append(ngrams(tweet, i))
            
        ngram_set = set(ngram_list)
        
        for ngram in ngram_set:
            try:
                new_tweet.append(ngram_dic[ngram])
            except KeyError:
                continue
        new_test_tweets.append(new_tweet)
        
    return new_train_tweets,   new_test_tweets


def load_embedding_matrix(filepath, EMB_DIM, word_index):
    """
    
    """
    embbeding_mat = np.zeros((len(word_index.keys()), EMB_DIM))
    embed_dict = {}
    with open(os.path.join(filepath, 'glove.twitter.27B.{}d.txt'.format(EMB_DIM)), 
              mode='rt', encoding='utf-8') as f:
        print("Loading embeddings...")
        for i, line in enumerate(f):
            key_vec = line.split()
            embed_dict.update({key_vec[0]:np.array(key_vec[1])})
            if i % 1e5 == 0:
                print("Loaded {:1.1E} words".format(i))
            
        print("Creating embedding matrix...")
        for word in word_index.keys():
            row = word_index[word]-1
            embbeding_mat[row, :] = word_index[word]
        
    return embbeding_mat

    
def main():
    
    DATA_PATH = "../data"
    TWEET_PATH = os.path.join(DATA_PATH, "twitter-datasets")
    FULL = True    
    EMB_DIM = 25
    NGRAM_RANGE = 2
    MAXLEN = 30

    
    train_tweets, test_tweets, labels = import_tweets(TWEET_PATH, FULL)
    train_tweets, test_tweets, word_index = tokenize(train_tweets, test_tweets)
    
    
    if NGRAM_RANGE:
        
        train_tweets_ngram, test_tweets_ngram = add_ngrams(train_tweets, test_tweets, len(word_index.keys()), ngram_range=NGRAM_RANGE)
        train_tweets_ngram = sequence.pad_sequences(train_tweets_ngram, maxlen=(MAXLEN*NGRAM_RANGE))
        test_tweets_ngram = sequence.pad_sequences(test_tweets_ngram, maxlen=(MAXLEN*NGRAM_RANGE))
        
        cPickle.dump([train_tweets_ngram, labels, test_tweets_ngram, len(word_index.keys())], 
                      open(os.path.join(DATA_PATH, 'train_test_{}_gram').format(NGRAM_RANGE), 'wb'))
        
    else:
        
        train_tweets= sequence.pad_sequences(train_tweets, maxlen=MAXLEN)
        test_tweets = sequence.pad_sequences(test_tweets, maxlen=MAXLEN)
        embedding_matrix = load_embedding_matrix(DATA_PATH, EMB_DIM, word_index)

        cPickle.dump([train_tweets, labels, test_tweets, len(word_index.keys()), embedding_matrix],
                      open(os.path.join(DATA_PATH, 'train_test_{}embedding'.format(EMB_DIM)), 'wb'))

if __name__ == '__main__':
    main()








































