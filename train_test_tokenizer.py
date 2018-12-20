import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import _pickle as cPickle
from nltk import ngrams
from collections import Counter


def import_tweets(filepath, FULL=False):
    """
    Imports the tweets from text files and returns them as a list of list of words
        
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
            test_tweets.append(' '.join(line.strip().split(',')[1:]))
            
    return train_tweets, test_tweets, labels
    
    
def tokenize(train_tweets, test_tweets, max_word=None):
    """
    Attribute a unique token value for in word from a list of tweets, 
    each tweet being a list of words. 
    IN: 
        train_tweets: list of list of words
        test_tweets: list of list of words
        max_word: integer, determines the number of most occuring different words to keep
    OUT:
        train_tweets_tokenized: list of list of tokens (int)
        test_tweets_tokenized: list of list of tokens (int)
        word_index: index linking words to corresponding token
    """
    if max_word == None:
        tokenizer = Tokenizer(filters='')
    else:
        tokenizer = Tokenizer(num_words=max_word, filters='')
    
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
        ngram_range: which ngrams to compute (bigram, trigram, etc.)
        min_occ: when not None, sets the minium of occurence of ngrams to keep
        max_occ: when not None, sets the maximum of occurence of ngrams to keep
        n_first: when not None, sets the number of most occuring ngrams to keep
    OUT: 
        ngram_dic: dictionary that has ngram of tokens for keys and an unique int for value
    '''
    # Creation of a list of all ngrams present in the tweet set provided
    ngram_list =[]
    for tweet in tweets:
        for i in range(2, ngram_range + 1):
            for n in ngrams(tweet, i):
                ngram_list.append(n)
    
    # counter of all ngrams to be able to filter them according to number of occurence        
    counter = Counter(ngram_list)
    # according to condition, extract the ngrams needed from the counter
    new_ngram_list = []
    if n_first:
        new_ngram_list = [val[0] for val in counter.most_common(n_first)]
    elif max_occ:
        new_ngram_list = [val[0] for val in counter.items() if val[1]>=min_occ and val[1]<=max_occ]
    else:
        new_ngram_list = [val[0] for val in counter.items() if val[1]>=min_occ]
    # create unique token indices for ngrams. Needs to be unique so we start after max_token
    new_tokens = range(max_token+1, len(new_ngram_list)+max_token+1)
    # creation of dictionary with ngram as key and token as value
    ngram_dic = dict(zip(new_ngram_list, new_tokens))
    
    return ngram_dic
        
def add_ngrams(tweet_list, ngram_dic, ngram_range=2):
    '''
    appends a token of the ngrams in the tweet. Tokens that are appended can be
    restricted in function of number of occurence.
    IN: 
        tweet_list: list of list of tokenized tweets
        ngram_dic: dictionary linking each ngram to a unique token
        ngram range: which ngram to compute (bigram, trigram, etc.)
    OUT: a list of list of tokens for each tweet with ngram tokens included
    '''
    
    new_train_tweets = []

    for tweet in tweet_list:
        ngram_list = []
        new_tweet = tweet
        for i in range(2, ngram_range + 1):
            for n in ngrams(tweet, i):
                ngram_list.append(n)
        ngram_set = set(ngram_list)
        
        for ngram in ngram_set:
            try:
                new_tweet.append(ngram_dic[ngram])
            except KeyError:
                continue
        new_train_tweets.append(new_tweet)
        
    return new_train_tweets


def load_embedding_matrix(filepath, EMB_DIM, word_index):
    """
    Loads the GloVe pretrained embedding matrix of dimension EMB_DIM and returns it
    as a numpy array.
    """
    embbeding_mat = np.zeros((len(word_index.keys()), EMB_DIM))
    embed_dict = {}
    with open(os.path.join(filepath, 'glove.twitter.27B.{}d.txt'.format(EMB_DIM)), 
              mode='rt', encoding='utf-8') as f:
        print("Loading embeddings...")
        for i, line in enumerate(f):
            key_vec = line.split()
            embed_dict.update({key_vec[0]:np.array(key_vec[1:])})
            if i % 1e5 == 0:
                print("Loaded {:1.1E} words".format(i))
            
        print("Creating embedding matrix...")
        for word in word_index.keys():
            row = word_index[word]-1
            try:
                embbeding_mat[row, :] = embed_dict[word]
            except KeyError:
                embbeding_mat[row, :] = np.zeros(EMB_DIM)
        
    return embbeding_mat

    
def main():
    
    DATA_PATH = "../data"
    TWEET_PATH = os.path.join(DATA_PATH, "twitter-datasets")
    FULL = True    
    EMB_DIM = 200
    NGRAM_RANGE = 3
    MAXLEN = 40
    
    n_first_ngram = None
    
    train_tweets, test_tweets, labels = import_tweets(TWEET_PATH, FULL)
    train_tweets, test_tweets, word_index = tokenize(train_tweets, test_tweets)
    max_word = len(word_index.keys())
    
    if NGRAM_RANGE:
        
        ngram_dic = create_ngram_dic(train_tweets, max_word, n_first=n_first_ngram)
        train_tweets_ngram = add_ngrams(train_tweets, ngram_dic, ngram_range=NGRAM_RANGE)
        test_tweets_ngram = add_ngrams(test_tweets, ngram_dic, ngram_range=NGRAM_RANGE)
        
        train_tweets_ngram = sequence.pad_sequences(train_tweets_ngram, maxlen=(MAXLEN*NGRAM_RANGE))
        test_tweets_ngram = sequence.pad_sequences(test_tweets_ngram, maxlen=(MAXLEN*NGRAM_RANGE))
        
        
        cPickle.dump([train_tweets_ngram, labels, test_tweets_ngram, max_word+len(ngram_dic)], 
                      open(os.path.join(DATA_PATH, 'for_graph_trigram.pkl').format(NGRAM_RANGE), 'wb'))
        
    else:
        
        train_tweets= sequence.pad_sequences(train_tweets, maxlen=MAXLEN)
        test_tweets = sequence.pad_sequences(test_tweets, maxlen=MAXLEN)
        embedding_matrix = load_embedding_matrix(DATA_PATH, EMB_DIM, word_index)
    
        cPickle.dump([train_tweets, labels, test_tweets, len(word_index.keys()), embedding_matrix],
                      open(os.path.join(DATA_PATH, 'for_graph_1gram.pkl'), 'wb'))

if __name__ == '__main__':
    main()








































