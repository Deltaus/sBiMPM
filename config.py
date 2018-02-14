#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 06:30:30 2018

@author: deltau
"""

class Config(object):
    
    # structure
    CHAR_LSTM_HIDDEN_SIZE = 15#20
    CHAR_EMBED_SIZE = 15#20
    BILSTM_HIDDEN_SIZE = 50#100
    WORD_INPUT_DIM = 50#100
    WORD_EMBED_SIZE = 100#300
    MAX_CHAR_NUM = 10#15
    MAX_WORD_NUM = 35#50
    BATCH_SIZE = 12#36
    NUM_OF_LSTMLAYERS = 2
    NUM_OF_EPOCH = 20
    MAX_VOCAB_SIZE = 10000
    # param
    PERSPECTS = 6
    DROPOUT = 0.1
    LR = 0.05
    FORGET_BIAS = 0.9
    MAX_GRAD_NORM = 5
    
    # temp
    embs = 300