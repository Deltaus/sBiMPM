#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:25:39 2018

@author: deltau
"""

import tensorflow as tf
from my_model import My_Model
import read
from config import Config
import train


if __name__=="__main__":
    
    path = 'snli_1.0/'
    #entire file
    #file_train = 'snli_1.0_train.jsonl'
    #file_valid = 'snli_1.0_dev.jsonl'
    #file_test =  'snli_1.0_test.jsonl'
    #sample file
    sample_file_train = 'snli_1.0_train.jsonl'
    sample_file_valid = 'snli_1.0_dev.jsonl'
    sample_file_test = 'snli_1.0_test.jsonl'
    
    train_summary = './summary'

    # word
    #train_data  labels in one-hot
    sent_pairs_train, embeddings_t = read.preprocess_data(path+sample_file_train, True, False)
    #vad_data
    sent_pairs_valid, _ = read.preprocess_data(path+sample_file_valid, True, False)
    #test_data
    sent_pairs_test, _ = read.preprocess_data(path+sample_file_test, True, False)
    
    #num of samples in each dataset
    train_sample_num = len(sent_pairs_train)
    train_iters = round(train_sample_num / Config.BATCH_SIZE)
    valid_sample_num = len(sent_pairs_valid)
    valid_iters = round(valid_sample_num / Config.BATCH_SIZE)
    test_sample_num = len(sent_pairs_test)
    test_iters = round(test_sample_num / Config.BATCH_SIZE)
    
    data = [sent_pairs_train, sent_pairs_valid, sent_pairs_test]
    iters = [train_iters, valid_iters, test_iters]
    log = [True, False, False]

    with tf.variable_scope('snli_model'):
        model = My_Model(is_training=True)
        model.tensor_calculate(Config)
        model.training()
#    with tf.variable_scope('snli_model', reuse=True):
#        eval_model = My_Model()
#        eval_model.tensor_calculate(Config)
    
    #merged = tf.summary_merge_all()
    #writer = tf.summary.FileWriter(train_summary, tf.get_default_graph())
    with tf.Session(config=tf.ConfigProto(device_count={"CPU":2})) as sess:
        #merged = tf.summary.merge_all()
        print 'start initialization'
        init = tf.global_variables_initializer()
        sess.run(init)
        train.epoch(sess, model, data, iters, log, summ=None, train_writer=None)
#        for epoch in range(Config.NUM_OF_EPOCH):
#            print 'In epoch: %d/%d' % (epoch + 1,Config.NUM_OF_EPOCH)
#            print 'Training:'
#            _, t_costs = train.run_epoch(sess, train_model, sent_pairs_train, train_iters, summ=None, train_op=train_model.train_op, train_writer=None, embed=embeddings_t, output_log=True)       
#            print 'Training total costs: %.3f' % t_costs
#            print 'Validating:'
#            e_acc, e_costs = train.run_epoch(sess, eval_model, sent_pairs_valid, valid_iters, summ=None, train_op=None, train_writer=None, embed=embeddings_t, output_log=False)
#            print 'Validate acc:%.3f, total costs: %.3f' % (e_acc,e_costs)
#        
#        print 'Testing:'
#        test_acc, test_costs = train.run_epoch(sess, eval_model, sent_pairs_test, test_iters, summ=None, train_op=None, train_writer=None, embed=embeddings_t, output_log=False)
#        print 'Test Accuracy: %.3f, Costs: %.3f' % (test_acc,test_costs)




