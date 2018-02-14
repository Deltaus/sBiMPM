#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:24:54 2018

@author: deltau
"""

import random
#import read
from config import Config
import numpy as np



def run_epoch(session, model, data, iters, summ=None, train_op=None, train_writer=None, output_log=False):
    
    total_costs = 0.0
    acc = 0
    iteration = int(iters)
    for i in range(iteration):
        prems, prems_char, hypos, hypos_char, labels = batch(data, Config.BATCH_SIZE)
        
        #print labels
        prems = np.array(prems)
        prems_char = np.array(prems_char)
        hypos = np.array(hypos)
        hypos_char = np.array(hypos_char)
        labels = np.array(labels)
        
        if train_op is not None:
            cost, acc, _ = session.run([model.cost, model.accuracy, train_op], 
                                       feed_dict={model.char_input_p:prems_char, 
                                                  model.word_input_p:prems, 
                                                  model.char_input_q:hypos_char, 
                                                  model.word_input_q:hypos, 
                                                  model.labels:labels})
        else:
            cost, acc = session.run([model.cost, model.accuracy], 
                                    feed_dict={model.char_input_p:prems_char, 
                                               model.word_input_p:prems, 
                                               model.char_input_q:hypos_char, 
                                               model.word_input_q:hypos, 
                                               model.labels:labels})                                 
        total_costs += cost  
        #train_writer.add_summary(summary, i)
        if output_log and i % 10 == 0:
            print 'After %d iteration(s), Acc is %.3f' % (i+1,acc)
            
    return acc, total_costs

def batch(data, batch_size):
    
    one_batch = random.sample(data, batch_size)
    prems = list()
    hypos = list()
    prems_char = list()
    hypos_char = list()
    labels = list()
    for p, h, pc, hc, l in one_batch:
        prems.append(p)
        hypos.append(h)
        prems_char.append(pc)
        hypos_char.append(hc)
        labels.append(l)
    
    return prems, prems_char, hypos, hypos_char, labels

 
def epoch(sess, model, data, iters, log, summ=None, train_writer=None):
    
#    model.training()
    
    for epoch in range(Config.NUM_OF_EPOCH):
        print 'In epoch: %d/%d' % (epoch + 1,Config.NUM_OF_EPOCH)
        print 'Training:'
        _, t_costs = run_epoch(sess, model, data[0], iters[0], summ=None, train_op=model.train_op, train_writer=None, output_log=log[0])       
        print 'Training total costs: %.3f' % t_costs
        print 'Validating:'
        e_acc, e_costs = run_epoch(sess, model, data[1], iters[1], summ=None, train_op=None, train_writer=None, output_log=log[1])
        print 'Validate acc:%.3f, total costs: %.3f' % (e_acc,e_costs)
        
    print 'Testing:'
    test_acc, test_costs = run_epoch(sess, model, data[2], iters[2], summ=None, train_op=None, train_writer=None, output_log=log[2])
    print 'Test Accuracy: %.3f, Costs: %.3f' % (test_acc,test_costs)
    
    