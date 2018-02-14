#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:25:16 2018

@author: deltau
"""

from nltk.tokenize import word_tokenize
from embedding import word_embedding as embed_vect
from embedding import char_embedding as char_ev
from collections import Counter
from config import Config
import numpy as np
import tensorflow as tf
import csv


def delete_bad(file_path):
    
    raw_pairs = list()
    with open(file_path,'r') as jsonl_file:
        for jsl in jsonl_file:
            raw_pairs.append(eval(jsl))
    
    #print len(raw_pairs)       
    bad_indexes = list()
    for raw in raw_pairs:
        if raw['gold_label'] == '-':
            bad_indexes.append(raw_pairs.index(raw))
    for ind in bad_indexes:
        if ind < len(raw_pairs):
            del raw_pairs[ind]
        
    return raw_pairs

def get_sentences(raw_pairs):
    
    sentence1s = list()
    sentence2s = list()
    labels = list()
    for raw in raw_pairs:
        if raw['gold_label'] != '-':
            sentence1s.append(raw['sentence1'].lower())
            sentence2s.append(raw['sentence2'].lower())
            labels.append(raw['gold_label'])
    
    #print sentence1s
    s1 = [word_tokenize(sentence) for sentence in sentence1s]  #word list of sentence1
    #print s1
    s2 = [word_tokenize(sentence) for sentence in sentence2s]  #word list of sentence2
    
    return s1, s2, labels

def pad_sentence(token_list, pad_length, pad_id):
    
    #padding = [pad_id] * (pad_length - len(token_list))
    if len(token_list) <= Config.MAX_WORD_NUM:
        padding = [pad_id] * (Config.MAX_WORD_NUM - len(token_list))
        padded_list = token_list + padding
    else:
        padded_list = token_list[:Config.MAX_WORD_NUM]
    
    return padded_list   #a padded list of a single sentence


def preprocess_data(file_path, is_read=False, is_write=False):
    
    raw_pairs = delete_bad(file_path)
    s1, s2, labels = get_sentences(raw_pairs) #s1: word list of lists of sentence1 / s2: ... / labels: list of labels

    categories = ["neutral", "entailment", "contradiction"]
    lab = list()
    for l in labels:
        onehot = [0,0,0]
        ind = categories.index(l)
        onehot[ind] = 1
        lab.append(onehot)       
    labels = lab
    
    #vocab & embedding processing
    sentences = list()
    for x in s1:
        sentences.append(x)
    for x in s2:
        sentences.append(x)
    
    # count word & write
    all_word_list = list()
    for sent in sentences:
            for word in sent:
                all_word_list.append(word)
    counter = Counter(all_word_list)
    count_pairs = sorted(counter.items(), key=lambda x : (-x[1], x[0]))
    wd, _ = list(zip(*count_pairs))
    wd = ["pad"] + ["unk"] + list(wd)
    word_to_id = dict(zip(wd[:Config.MAX_VOCAB_SIZE], range(Config.MAX_VOCAB_SIZE)))
    
    with open('./vocab/vocab.txt', "w") as file:
        for word, id in word_to_id.items():
            file.write("{}\t{}\n".format(word,id))
    
    # whether read embeddings or generate them
    if is_read:
        embeddings = {}
        with open('./embedding/embed.csv', 'rt') as csvfile:
            r = csv.reader(csvfile)
            for row in r:
                embeddings[row[0]] = np.array(row[1:Config.WORD_EMBED_SIZE+1]).tolist()
    else:
        sentences, embeddings = embed_vect(sentences)  #return filtered sentence list and embedding model
    
    # whether write embeddings or not
    if is_write:            
        ipt = list()
        for word in word_to_id:
            words = list()
            words.append(word)
            try:
                for vect in np.array(embeddings[word]).tolist():
                    words.append(vect)
            except:
                tv = np.random.uniform(size=Config.WORD_EMBED_SIZE).tolist()
                for vect in tv:
                    words.append(vect)
            ipt.append(words)        
                
        with open("./embedding/embed.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            #先写入columns_name
            for row in ipt:
                writer.writerow(row)
  
    max_prem_len = Config.MAX_WORD_NUM
    max_hypo_len = Config.MAX_WORD_NUM
    prem = [pad_sentence(x, max_prem_len, '<pad>') for x in s1]
    hypo = [pad_sentence(x, max_hypo_len, '<pad>') for x in s2]
    
    # process word & char embed
    prems, prems_char = get_embeddings(prem, embeddings)
    hypos, hypos_char = get_embeddings(hypo, embeddings)

    #train_data
    sent_pairs = zip(prems, hypos, prems_char, hypos_char, labels)
    #char_pairs = list(zip(prems_char, hypos_char))
    
    return sent_pairs, embeddings #, all_word_dict

def get_embeddings(inputs, embedding):
    
    #inputs 
    char_emb = char_ev()
    
    ipt = list()
    ipt_char = list()
    for sent in inputs:
        ssent = sent[:Config.MAX_WORD_NUM]
        words = list()
        words_char = list()
        for word in ssent:
            if word == '<pad>':
                words.append(np.zeros([Config.WORD_EMBED_SIZE])[:Config.WORD_EMBED_SIZE].tolist())
                words_char.append(np.zeros([Config.MAX_CHAR_NUM, Config.CHAR_EMBED_SIZE])[:Config.MAX_CHAR_NUM,:Config.CHAR_EMBED_SIZE].tolist())
            else:
                try:
                    words.append(np.array(embedding[word])[:Config.WORD_EMBED_SIZE].tolist())
                    char_list = [ char_emb[x] for x in list(word)]
                    if len(char_list) > Config.MAX_CHAR_NUM:
                        char_list = char_list[:Config.MAX_CHAR_NUME]
                    else:
                        pad = [ x * 0 for x in range(Config.CHAR_EMBED_SIZE)]
                        pad = [pad for i in  range(Config.MAX_CHAR_NUM - len(char_list))]
                        [char_list.append(x) for x in pad]
                    words_char.append(char_list[:Config.MAX_CHAR_NUME])
                except:
                    words.append(np.random.uniform(size=(Config.WORD_EMBED_SIZE))[:Config.WORD_EMBED_SIZE].tolist())
                    words_char.append(np.random.uniform(size=(Config.MAX_CHAR_NUM, Config.CHAR_EMBED_SIZE))[:Config.MAX_CHAR_NUM,:Config.CHAR_EMBED_SIZE].tolist())
        ipt.append(words[:Config.MAX_WORD_NUM])
        ipt_char.append(words_char)

    return ipt, ipt_char

def write_embedding(embedding):
    
    vocab = []
    with open('/Users/deltau/Downloads/Decomposable_Attn-master/vocab/vocab.txt','r') as vfile:
        line = vfile.readline()
        while line:
            word = line.split()[0]
            #word = word.split('\\')[0]
            vocab.append(word)
            line  =vfile.readline()
    
    
       
    ipt = list()
    for word in vocab:
        words = list()
        words.append(word)
        try:
            for vect in embedding[word].tolist():
                words.append(vect)
        except:
            tv = np.random.uniform(size=Config.WORD_EMBED_SIZE).tolist()
            for vect in tv:
                words.append(vect)
        ipt.append(words)        
                
    with open("/Users/deltau/Downloads/Decomposable_Attn-master/embedding/embed.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        for row in ipt:
            writer.writerow(row)
    
    
    
    
    





