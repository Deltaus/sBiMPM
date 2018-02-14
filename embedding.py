#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 05:38:44 2018

@author: deltau
"""

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from config import Config as conf
import numpy as np

punct = ['.',',','?','(',')','/','\'','\\',':',';','<','>','-']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def word_embedding(token_lists): #input sentences
    
    fillist = []
    for token_list in token_lists:
        filtered = []
        for word in token_list:
            if not word in stopwords.words('english') and not word in punct:
                filtered.append(word)
        fillist.append(filtered)

    token_lists = fillist
    model = Word2Vec(token_lists, sg=1, size=conf.WORD_EMBED_SIZE, window=4, min_count=1)

    return token_lists, model

def char_embedding():
    
    char_embed = dict()
    for char in alphabet:
        char_embed[char] = np.random.normal(size=conf.CHAR_EMBED_SIZE).tolist()
        
    return char_embed