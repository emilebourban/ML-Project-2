# -*- coding: utf-8 -*-

import os

def load_dicts(Dict_PATH):
    dict_emnlp = {}
    with open(os.path.join(DICT_PATH, "emnlp_dict.txt"), mode='rt') as f:
        for line in f:
            key, value = line.rstrip('\n').split('\t')
            dict_emnlp.update({key:value})
            
    dict_pairs = {}
    with open(os.path.join(DICT_PATH, "Test_Set_3802_Pairs.txt") , mode='r') as f:
        for line in f:
            try:
                key, value = line.rstrip('\n').split('\t')[1].split(' | ')
                dict_pairs.update({key:value})
            # Some values have multiple keys affected to them
            except ValueError:
                ls = line.rstrip('\n').split('\t')[1].split(' | ')
                value = ls[-1]
                ls_keys = ls[:-1]
                    # Update dict with all the keys
                for key in ls_keys:
                    dict_pairs.update({key:value})
                    
    dict_typos = {}
    with open(os.path.join(DICT_PATH,  "typo-corpus-r1.txt"), mode='rt') as f:
        for line in f:
            key, value, _, _, _, _ = line.rstrip('\n').split('\t')
            dict_typos.update({key:value})
           
    return dict_pairs, dict_typos, dict_emnlp

def clean_typos(filename, in_path, out_path):
    with open(os.path.join(in_path, filename), mode='rt', encoding='utf-8') as rf:
        with open(os.path.join(out_path, 'cl_'+filename), mode='wt', encoding='utf-8') as wf:
            for line in rf:
                    twitt = line.rstrip('\n').split(' ')
                    for i, word in enumerate(twitt):
                        if word in dict1.keys():
                            twitt[i] = dict1[word]     
                        elif word in dict2.keys():
                            twitt[i] = dict2[word]  
                        elif word in dict3.keys():
                            twitt[i] = dict3[word]  
    
                    wf.write(' '.join(twitt)+'\n')

DICT_PATH = "../dict"
TWITT_PATH = "../twitter-datasets"
DATA_PATH = "../data"

dict1, dict2, dict3 = load_dicts(DICT_PATH)

files = [i for i in os.listdir(TWITT_PATH) if not i.startswith('vocab') and i.endswith('.txt')]

for file in files:
    clean_typos( file, TWITT_PATH, os.path.join(TWITT_PATH, "clean"))
                    






















