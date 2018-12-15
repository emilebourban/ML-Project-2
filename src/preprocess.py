# -*- coding: utf-8 -*-

import os
import re
from nltk.stem.porter import *

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
                value = ls[-1]
                ls_keys = ls[:-1]
                    # Update dict with all the keys
                for key in ls_keys:
                    dict_typo.update({key:value})
                    
    with open(os.path.join(DICT_PATH,  "typo-corpus-r1.txt"), mode='rt') as f:
        for line in f:
            key, value, _, _, _, _ = line.rstrip('\n').split('\t')
            dict_typo.update({key:value})
           
    return dict_typo


def clean_tweets(filename, in_path, out_path, dict_typo, only_words=False, stemming=False, min_len=None):
    """
    
    """
    with open(os.path.join(in_path, filename), mode='rt', encoding='utf-8') as rf:
        with open(os.path.join(out_path, 'cl_'+filename), mode='wt', encoding='utf-8') as wf:            
            for line in rf:
                
                tweet = re.sub(r'<([^>]+)>', ' ', line.strip().lower())         # Removes usr and url
                tweet = re.sub(r'^#| #', ' ', tweet)                            # Removes hashtags
                tweet = re.sub(r'\d+(x)\d+', '<img>', tweet)                    # Removes picture frames            
                tweet = re.sub(r'n\'t$', ' not', tweet)                         # Converts negation contraction to verb + not*  
                
                if only_words:
                    tweet = re.sub(r'[^a-z]', ' ', tweet)                       # Only keeps words
                    
                tweet = tweet.strip().split()
                
                if stemming:
                    stemmer = PorterStemmer()
                    tweet = [stemmer.stem(word) for word in tweet]              # stemming
                
                for i, word in enumerate(tweet):        
                    if word in dict_typo.keys() and word != 'not':
                        tweet[i] = dict_typo[word]                     
                
                if min_len is not None:
                    wf.write(' '.join([word for word in tweet if len(word) >= min_len])+'\n')
                else:
                    wf.write(' '.join(tweet)+'\n')

def main():

    DICT_PATH = "../dict"
    OR_TWITT_PATH = "../data/twitter-datasets-original"
    NEW_TWITT_PATH = "../data/twitter-datasets"
    DATA_PATH = "../data"
    FULL = True 
    
    dict_typo = load_dicts(DICT_PATH)
    
    if FULL:
        files = [i for i in os.listdir(OR_TWITT_PATH) if i.endswith('.txt')]        
    else:
        files = [i for i in os.listdir(OR_TWITT_PATH) if not i.endswith('full.txt')]
    
    for file in files:
        print("Processing {} ...".format(file))
        clean_tweets(file, OR_TWITT_PATH, NEW_TWITT_PATH, dict_typo, only_words=True, stemming=False, min_len=3)
                        
if __name__ == '__main__':
    main()





















