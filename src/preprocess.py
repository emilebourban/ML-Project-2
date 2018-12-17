# -*- coding: utf-8 -*-

import os
import re
from nltk.stem.porter import *
from nltk.corpus import brown
import itertools

def load_dicts(DICT_PATH):
    """
    
    """
    dict_typo = {}
    with open(os.path.join(DICT_PATH, "emnlp_dict.txt"), mode='rt') as f:
        for line in f:
            key, value = line.rstrip('\n').split('\t')
            dict_typo.update({key:value})

    with open(os.path.join(DICT_PATH, "Test_Set_3802_Pairs.txt") , mode='r') as f:
        for line in f:
            try:
                key, value = line.rstrip('\n').split('\t')[1].split(' | ')
                dict_typo.update({key:value})
            # Some values have multiple keys affected to them
            except ValueError:
                ls = line.rstrip('\n').split('\t')[1].split(' | ')
                key = ls[0]
                value= ls[1]
                    # Update dict with all the keys
                dict_typo.update({key:value})
                    
    with open(os.path.join(DICT_PATH,  "typo-corpus-r1.txt"), mode='rt') as f:
        for line in f:
            key, value, _, _, _, _ = line.rstrip('\n').split('\t')
            dict_typo.update({key:value})
           
    return dict_typo


def remove_repetitions(tweet):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)
    
    """
    tweet = tweet.split()
    for i in range(len(tweet)):
#        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
#        if len(tweet[i])>0:
#            if tweet[i] not in word_list:
        tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet


def clean_tweets(filename, in_path, out_path, dict_typo, word_list, only_words=False, stemmer=None, min_len=None):
    """
    
    """
    print("Cleaning with: only words={}, stemmer={}, minimal length={}".format(only_words, stemmer!=None, min_len))    
    with open(os.path.join(in_path, filename), mode='rt', encoding='utf-8') as rf:
        with open(os.path.join(out_path, 'cl_'+filename), mode='wt', encoding='utf-8') as wf:
            
            for line in rf:
                if 'test' in filename:
                    ID = line.strip().split(',')[0]+','
                    tweet = ' '.join(line.strip().split()[1:])
                else:
                    ID = ''
                    tweet =  line.strip()
                    
                remove_repetitions(tweet)
                
                tweet = tweet.strip().split()
                for i, word in enumerate(tweet):        
                    if word in dict_typo.keys():
                        tweet[i] = dict_typo[word]  
                        
                tweet = ' '.join(tweet)
                    
                tweet = re.sub(r"\'s", " \'s", tweet)
                tweet = re.sub(r"\'ve", " \'ve", tweet) 
                tweet = re.sub(r"n\'t", " n\'t", tweet)
                tweet = re.sub(r" ca ", " can ", tweet)
                tweet = re.sub(r"\'re", " \'re", tweet)
                tweet = re.sub(r"\'d", " \'d", tweet)
                tweet = re.sub(r"\'l", " \'ll", tweet)
                tweet = re.sub(r"\'ll", " \'ll", tweet)
#                tweet = re.sub(r",", " , ", tweet)
#                tweet = re.sub(r"!", " ! ", tweet)
#                tweet = re.sub(r"\(", " \( ", tweet)
#                tweet = re.sub(r"\)", " \) ", tweet)
#                tweet = re.sub(r"\?", " \? ", tweet)
                tweet = re.sub(r"\s{2,}", " ", tweet)
                tweet = re.sub(r'<([^>]+)>', ' ',tweet)         # Removes usr and url
                tweet = re.sub(r'^#| #', ' ', tweet)                            # Removes hashtags
                tweet = re.sub(r'\d+(x)\d+', '<img>', tweet)                    # Removes picture frames            
#                tweet = re.sub(r'n\'t$|n\'t ', ' not', tweet)                   # Converts negation contraction to verb + not
                
                if only_words:
                    tweet = re.sub(r'[^a-z]', ' ', tweet)                       # Only keeps words
                 
                tweet = tweet.strip().split()
                
                if stemmer != None:
                    tweet = [stemmer.stem(word) for word in tweet]              # stemming
                
                # Spell checker for commonly missspeled words
                for i, word in enumerate(tweet):        
                    if word in dict_typo.keys():
                        tweet[i] = dict_typo[word]                     
                
                if min_len is not None:
                    wf.write(ID+' '.join([word for word in tweet if len(word) >= min_len])+'\n')
                else:
                    wf.write(ID+' '.join(tweet)+'\n')
                    

def main():

    DICT_PATH = "../dict"
    OR_TWITT_PATH = "../data/twitter-datasets-original"
    NEW_TWITT_PATH = "../data/twitter-datasets"
    DATA_PATH = "../data"
    FULL = True 
#    
    dict_typo = load_dicts(DICT_PATH)
    word_list = brown.words()

    if FULL:
        files = [i for i in os.listdir(OR_TWITT_PATH) if i.endswith('.txt')]        
    else:
        files = [i for i in os.listdir(OR_TWITT_PATH) if not i.endswith('full.txt')]
    
    stemmer = PorterStemmer()
    
    for file in files:
        print("Processing {} ...".format(file))
        clean_tweets(file, OR_TWITT_PATH, NEW_TWITT_PATH, dict_typo, word_list, only_words=False, stemmer=None, min_len=None)
                        
if __name__ == '__main__':
    main()





















