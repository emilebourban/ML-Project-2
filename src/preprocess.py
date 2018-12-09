# -*- coding: utf-8 -*-

import os
import re

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


def clean_tweets(filename, in_path, out_path, dict_typo):
    """
    
    """
    # Removes all the common words that don't carry sense
    del_list = ['<user>', '<url>', 'to', 'the', 'my', 'it', 'and', 'you', 'is', 'in', 'for']
    with open(os.path.join(in_path, filename), mode='rt', encoding='utf-8') as rf:
        with open(os.path.join(out_path, 'cl_'+filename), mode='wt', encoding='utf-8') as wf:            
            for line in rf:
            
                tweet = re.sub(r'n\'t$', ' not', line)          # Converts negation contraction to verb + not
                tweet = re.sub(r'\d+(x)\d+|^\d+(x)\d+ | \d+(x)\d+$', '', tweet)        # Removes picture frames
                tweet = re.sub(r'^#| #', '', tweet)        # Removes hashtags
#                tweet = re.sub(r'[^a-z ]', '', tweet)       # Only keeps words
                tweet = tweet.strip().split()
                tweet = [w for w in tweet if (w not in del_list)]
                for i, word in enumerate(tweet):                    
                    if word in dict_typo.keys():
                        tweet[i] = dict_typo[word] 
                    
                wf.write(' '.join(tweet)+'\n')

def main():

    DICT_PATH = "../dict"
    OR_TWITT_PATH = "../data/twitter-datasets-original"
    NEW_TWITT_PATH = "../data/twitter-datasets"
    DATA_PATH = "../data"
    FULL = False
    
    dict_typo = load_dicts(DICT_PATH)
    
    if FULL:
        files = [i for i in os.listdir(OR_TWITT_PATH) if i.endswith('full.txt')]        
    else:
        files = [i for i in os.listdir(OR_TWITT_PATH) if not i.endswith('full.txt')]
    
    for file in files:
        print("Processing {} ...".format(file))
        clean_tweets(file, OR_TWITT_PATH, NEW_TWITT_PATH, dict_typo)
                        
if __name__ == '__main__':
    main()





















