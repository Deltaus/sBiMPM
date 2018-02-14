#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 06:23:55 2018

@author: deltau
"""

import tensorflow as tf
from config import Config as conf
import time

class My_Model(object):
    
    def __init__(self, is_training=False):
        
        init_ts = time.time()
        self.char_lstm_hidden_size = conf.CHAR_LSTM_HIDDEN_SIZE
        self.lstm_hidden_size = conf.BILSTM_HIDDEN_SIZE
        self.is_training = is_training
        # words of sentence p and chars in it
        self.char_input_p = tf.placeholder(tf.float32, [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
        self.word_input_p = tf.placeholder(tf.float32, [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.WORD_EMBED_SIZE])
        # words of sentence q and chars in it
        self.char_input_q = tf.placeholder(tf.float32, [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
        self.word_input_q = tf.placeholder(tf.float32, [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.WORD_EMBED_SIZE])
        # labels
        self.labels = tf.placeholder(tf.int32, [None, 3])
       
        init_te = time.time()
        print 'init time: ', (init_te-init_ts)
#        # word representation 
#        #output_p1, output_fw_p1, output_bw_p1 = self.BiLSTM_builder(self.char_lstm_hidden_size, conf.FORGET_BIAS, self.char_input_p, 'word_represent_p' )
#        #output_q1, output_fw_q1, output_bw_q1 = self.BiLSTM_builder(self.char_lstm_hidden_size, conf.FORGET_BIAS, self.char_input_q, 'word_represent_q' )
#        char_p = self.LSTM_builder(conf.CHAR_LSTM_HIDDEN_SIZE, conf.DROPOUT, self.char_input_p, conf.MAX_CHAR_NUM, 'lstm' )  # BAS x MAX_W x MAX_C x 20
#        char_q = self.LSTM_builder(conf.CHAR_LSTM_HIDDEN_SIZE, conf.DROPOUT, self.char_input_q, conf.MAX_CHAR_NUM, 'lstm' )  # BAS x MAX_W x MAX_C x 20
#        
#        # word embedding and char composed embedding projection
#        char_p = tf.reshape(tf.convert_to_tensor(char_p), [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
#        char_p = [tf.squeeze(x) for x in tf.split(char_p, conf.MAX_CHAR_NUM, 2)]
#        char_p = tf.concat(char_p, 2)  # BAS x MAX_W x 300
#        #print char_p.shape
#        
#        char_q = tf.reshape(tf.convert_to_tensor(char_q), [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
#        char_q = [tf.squeeze(x) for x in tf.split(char_q, conf.MAX_CHAR_NUM, 2)]
#        char_q = tf.concat(char_q, 2)  # BAS x MAX_W x 300
#        #print char_q.shape
#        
#        final_word_embed_p = tf.concat([self.word_input_p, char_p], 2) # BAS x MAX_W x 600
#        final_word_embed_q = tf.concat([self.word_input_q, char_q], 2) # BAS x MAX_W x 600
#        
#        final_word_embed_p = self.projection(final_word_embed_p, 'proj')  # BAS x MAX_W x 100
#        final_word_embed_q = self.projection(final_word_embed_q, 'proj')  # BAS x MAX_W x 100
#        
#        # context representation
#        #output MAX_W x BAS x 100*2  output_fw BAS x 100  output_bw BAS x 100
#        output_p1, output_fw_p1, output_bw_p1 = self.BiLSTM_builder(self.lstm_hidden_size, conf.FORGET_BIAS, final_word_embed_p, 'context_represent' )
#        output_q1, output_fw_q1, output_bw_q1 = self.BiLSTM_builder(self.lstm_hidden_size, conf.FORGET_BIAS, final_word_embed_q, 'context_represent' )
#        
#        # matching
#        #p
#        output_p1 = tf.transpose(output_p1, [1,0,2]) # BAS x MAX_W x 100*2
#        output_p1_f = output_p1[:, :, :conf.BILSTM_HIDDEN_SIZE] # BAS x MAX_W x 100
#        output_p1_b = output_p1[:, :, conf.BILSTM_HIDDEN_SIZE:] # BAS x MAX_W x 100
#        
#        #q
#        output_q1 = tf.transpose(output_q1, [1,0,2]) # BAS x MAX_W x 100*2
#        output_q1_f = output_q1[:, :, :conf.BILSTM_HIDDEN_SIZE] # BAS x MAX_W x 100
#        output_q1_b = output_q1[:, :, conf.BILSTM_HIDDEN_SIZE:] # BAS x MAX_W x 100
#        
#        # p to q 
#        if method == 0:
#            output_fw_q1 = output_fw_q1[1]
#            output_fw_q1 = tf.tile(tf.expand_dims(output_fw_q1,1), [1, conf.MAX_WORD_NUM, 1]) #BAS x MAX_W x 100
#            m_f = self.fully_match(output_p1_f, output_fw_q1)   # BAS x MAX_W x l
#            output_bw_q1 = output_bw_q1[1]
#            output_bw_q1 = tf.tile(tf.expand_dims(output_bw_q1,1), [1, conf.MAX_WORD_NUM, 1])
#            m_b = self.fully_match(output_p1_b, output_bw_q1)
#        elif method == 1:
#            m_f = self.max_pooling()
#            m_b = self.max_pooling()
#        elif method == 2:
#            m_f = self.attentive_match()
#            m_b = self.attentive_match()
#        elif method == 3:
#            m_f = self.max_attentive()
#            m_b = self.max_attentive()
#        else:
#            print 'No such method, change to method 1...'
#            m_f = self.fully_match()
#            m_b = self.fully_match()
#        
#        # aggregate
#        _, output_fw_p2, output_bw_p2 = self.BiLSTM_builder(conf.PERSPECTS, conf.FORGET_BIAS, m_f, 'aggregate' )
#        _, output_fw_q2, output_bw_q2 = self.BiLSTM_builder(conf.PERSPECTS, conf.FORGET_BIAS, m_b, 'aggregate' )
#        
#        concat_vect =tf.reshape( tf.concat([output_fw_p2[1],output_bw_p2[1], output_fw_q2[1], output_bw_q2[1]],1), [conf.BATCH_SIZE, -1])
#        self.categ = self.feed_forword(concat_vect, conf.MAX_WORD_NUM*4, 'output')
#        
#        _, labels = tf.nn.top_k(self.labels) # output: value, indices
#        _, categ = tf.nn.top_k(self.categ)
#        
#        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.categ], [labels], [tf.ones([conf.BATCH_SIZE])])
#        self.cost = tf.reduce_mean(loss) 
#        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(categ, labels), tf.float32))
#        
#        # training
#        if is_training:
#            
#            print 'training setting'
#            self.lr = tf.Variable(conf.LR, trainable=False)
#            tvars = tf.trainable_variables()
#            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), conf.MAX_GRAD_NORM)
#            optimizer = tf.train.AdagradOptimizer(self.lr)          
#            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
#        if self.is_training:
#            self.training()
#    #########################################################################
    
    def tensor_calculate(self, conf, method=0):
        
        # word representation 
        with tf.name_scope('LSTM_word_represent'):
            #output_p1, output_fw_p1, output_bw_p1 = self.BiLSTM_builder(self.char_lstm_hidden_size, conf.FORGET_BIAS, self.char_input_p, 'word_represent_p' )
            #output_q1, output_fw_q1, output_bw_q1 = self.BiLSTM_builder(self.char_lstm_hidden_size, conf.FORGET_BIAS, self.char_input_q, 'word_represent_q' )
            char_p = self.LSTM_builder(conf.CHAR_LSTM_HIDDEN_SIZE, conf.DROPOUT, self.char_input_p, conf.MAX_CHAR_NUM, 'lstm' )  # BAS x MAX_W x MAX_C x 20
            char_q = self.LSTM_builder(conf.CHAR_LSTM_HIDDEN_SIZE, conf.DROPOUT, self.char_input_q, conf.MAX_CHAR_NUM, 'lstm' )  # BAS x MAX_W x MAX_C x 20
        
            # word embedding and char composed embedding projection
            char_p = tf.reshape(tf.convert_to_tensor(char_p), [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
            char_p = [tf.squeeze(x) for x in tf.split(char_p, conf.MAX_CHAR_NUM, 2)]
            char_p = tf.concat(char_p, 2)  # BAS x MAX_W x 300
            #print char_p.shape
        
            char_q = tf.reshape(tf.convert_to_tensor(char_q), [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.MAX_CHAR_NUM, conf.CHAR_EMBED_SIZE])
            char_q = [tf.squeeze(x) for x in tf.split(char_q, conf.MAX_CHAR_NUM, 2)]
            char_q = tf.concat(char_q, 2)  # BAS x MAX_W x 300
            #print char_q.shape
            
            with tf.name_scope('word_embed_proj'):
                final_word_embed_p = tf.concat([self.word_input_p, char_p], 2) # BAS x MAX_W x 600
                final_word_embed_q = tf.concat([self.word_input_q, char_q], 2) # BAS x MAX_W x 600
        
                final_word_embed_p = self.projection(final_word_embed_p, 'proj')  # BAS x MAX_W x 100
                final_word_embed_q = self.projection(final_word_embed_q, 'proj')  # BAS x MAX_W x 100
        
        # context representation
        with tf.name_scope('BiLSTM_context_represent'):
            #output MAX_W x BAS x 100*2  output_fw BAS x 100  output_bw BAS x 100
            output_p1, output_fw_p1, output_bw_p1 = self.BiLSTM_builder(self.lstm_hidden_size, conf.FORGET_BIAS, final_word_embed_p, 'context_represent' )
            output_q1, output_fw_q1, output_bw_q1 = self.BiLSTM_builder(self.lstm_hidden_size, conf.FORGET_BIAS, final_word_embed_q, 'context_represent' )
            
        # matching
        with tf.name_scope('matching'):
            #p
            output_p1 = tf.transpose(output_p1, [1,0,2]) # BAS x MAX_W x 100*2
            output_p1_f = output_p1[:, :, :conf.BILSTM_HIDDEN_SIZE] # BAS x MAX_W x 100
            output_p1_b = output_p1[:, :, conf.BILSTM_HIDDEN_SIZE:] # BAS x MAX_W x 100
            
            #q
            output_q1 = tf.transpose(output_q1, [1,0,2]) # BAS x MAX_W x 100*2
            output_q1_f = output_q1[:, :, :conf.BILSTM_HIDDEN_SIZE] # BAS x MAX_W x 100
            output_q1_b = output_q1[:, :, conf.BILSTM_HIDDEN_SIZE:] # BAS x MAX_W x 100
            
            # p to q 
            if method == 0:
                output_fw_q1 = output_fw_q1[1]
                output_fw_q1 = tf.tile(tf.expand_dims(output_fw_q1,1), [1, conf.MAX_WORD_NUM, 1]) #BAS x MAX_W x 100
                m_f = self.fully_match(output_p1_f, output_fw_q1)   # BAS x MAX_W x l
                output_bw_q1 = output_bw_q1[1]
                output_bw_q1 = tf.tile(tf.expand_dims(output_bw_q1,1), [1, conf.MAX_WORD_NUM, 1])
                m_b = self.fully_match(output_p1_b, output_bw_q1)
            elif method == 1:
                m_f = self.max_pooling()
                m_b = self.max_pooling()
            elif method == 2:
                m_f = self.attentive_match()
                m_b = self.attentive_match()
            elif method == 3:
                m_f = self.max_attentive()
                m_b = self.max_attentive()
            else:
                print 'No such method, change to method 1...'
                m_f = self.fully_match()
                m_b = self.fully_match()
        
        # aggregate
        with tf.name_scope('BiLSTM_aggregate'):
            _, output_fw_p2, output_bw_p2 = self.BiLSTM_builder(conf.PERSPECTS, conf.FORGET_BIAS, m_f, 'aggregate' )
            _, output_fw_q2, output_bw_q2 = self.BiLSTM_builder(conf.PERSPECTS, conf.FORGET_BIAS, m_b, 'aggregate' )
            
            concat_vect =tf.reshape( tf.concat([output_fw_p2[1],output_bw_p2[1], output_fw_q2[1], output_bw_q2[1]],1), [conf.BATCH_SIZE, -1])
            self.categ = self.feed_forword(concat_vect, conf.MAX_WORD_NUM*4, 'output')
            
            _, labels = tf.nn.top_k(self.labels) # output: value, indices
            _, categ = tf.nn.top_k(self.categ)
            
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.categ], [labels], [tf.ones([conf.BATCH_SIZE])])
            self.cost = tf.reduce_mean(loss) 
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(categ, labels), tf.float32))
        
   
    def training(self): 
        
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            trainset_ts = time.time()
            print 'training setting'
            self.lr = tf.Variable(conf.LR, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), conf.MAX_GRAD_NORM)
            optimizer = tf.train.AdagradOptimizer(self.lr)          
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            
            trainset_te = time.time()
            print 'trainset time: ', (trainset_te-trainset_ts)
    
    ###########################################################################    
                         #CHAR_LSTM_HIDDEN_SIZE 20       #MAX_CHAR_NUM 15
    def LSTM_builder(self, hidden_size, dropout, inputs, steps, name):
        
        lstm_ts = time.time()
        print 'start building LSTM'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            if self.is_training:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*conf.NUM_OF_LSTMLAYERS)
            
        state = cell.zero_state(conf.MAX_WORD_NUM, tf.float32)
        inputs = [tf.squeeze(x) for x in tf.split(inputs, conf.BATCH_SIZE, 0)] # a list of tensors  BAS x MAX_W x MAX_C x 20
        outputs = []
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            for sent in inputs:                                           # MAX_W x MAX_C x 20
                cell_outputs = []
                for step in range(steps):                                 # MAX_W x 20
                    if step>0: tf.get_variable_scope().reuse_variables()
                    cell_output, state = cell(tf.squeeze(sent[:, step, :]), state)
                    cell_outputs.append(cell_output)
                outputs.append(cell_outputs)
        
        lstm_te = time.time()
        
        print 'completed...,time: ', (lstm_te - lstm_ts)        
        return outputs  # 4D list
    
    
    def BiLSTM_builder(self, hidden_size, forget_bias, inputs, name):
        
        bilstm_ts = time.time()
        print 'start building BiLSTM'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias)
            
            inputs = [tf.squeeze(x) for x in tf.split(inputs, conf.MAX_WORD_NUM, 1)]
            #output MAX_W x BAS x 100*2  output_fw BAS x 100  output_bw BAS x 100
            output, output_fw, output_bw = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32) 
        
        bilstm_te = time.time()
        print 'completed..., time: ', (bilstm_te - bilstm_ts)    
        return output, output_fw, output_bw
        
    
    def projection(self, inputs, name):
        
        print 'start projection from 600 to 100'
        # inputs: BAS x MAX_W x 600
        steps = inputs.get_shape()[1] #MAX_W
        hidden_size = inputs.get_shape()[2] #600
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w_proj', [conf.BATCH_SIZE, hidden_size, conf.BILSTM_HIDDEN_SIZE]) # BAS x 600 x 100
            b = tf.get_variable('b_proj', [steps, conf.BILSTM_HIDDEN_SIZE]) # MAX_W x 100
            
            reduced_embeddings = tf.matmul(inputs, w) + tf.tile(tf.expand_dims(b, 0), [conf.BATCH_SIZE, 1, 1])# BAS x MAX_W x 100
        
        print 'completed...' 
        return reduced_embeddings
 
    
    def multi_f(self, v1, v2, name):
        
        multi_ts = time.time()
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w_f = tf.get_variable('w_f', [conf.PERSPECTS, conf.BILSTM_HIDDEN_SIZE]) # l x d
            
             # make v1 , v2 BAS x MAX_W x d 
             #print 'v1: ', v1.shape
            v1 = tf.reshape(v1, [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W) x d
            v2 = tf.reshape(v2, [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W) x d
            v1 = tf.reshape(tf.tile(v1, [1, conf.PERSPECTS]) , [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W x l) x d
            v2 = tf.reshape(tf.tile(v2, [1, conf.PERSPECTS]) , [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W x l) x d
            v1 = [ tf.squeeze(x) for x in tf.split(v1, conf.BATCH_SIZE * conf.MAX_WORD_NUM, 0)]
            v2 = [ tf.squeeze(x) for x in tf.split(v2, conf.BATCH_SIZE * conf.MAX_WORD_NUM, 0)]
            for i1,i2 in zip(v1,v2):
                i1 = tf.nn.l2_normalize(w_f * i1, 1)  # l x d   
                i2 = tf.nn.l2_normalize(w_f * i2, 1)  # l x d
            v1 = tf.reshape(v1, [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W x l) x d
            v2 = tf.reshape(v2, [-1, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W x l) x d
            v = v1 * v2 # (BAS x MAX_W x l) x d
            v = tf.reshape(v, [-1, conf.PERSPECTS, conf.BILSTM_HIDDEN_SIZE]) # (BAS x MAX_W) x l x d
            final = tf.reduce_sum(v, 2) 
        
        multi_te = time.time() 
        print 'multi_p time:  ',(multi_te - multi_ts)          
        return tf.reshape(final, [conf.BATCH_SIZE, conf.MAX_WORD_NUM, conf.PERSPECTS])  # BAS x MAX_W x l
        
        
    def fully_match(self, input1, input2):
        
        return self.multi_f(input1, input2, 'fully_match')
     
        
    def max_pooling(self):
        pass
     
        
    def attentive_match(self):
        pass
    
    
    def max_attentive(self):
        pass
       
        
    def feed_forword(self, inputs, hidden_size, name):
        
        print 'feed forward network building'
        batch = inputs.get_shape()[0]
        hidden = inputs.get_shape()[1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable('w1_ff', [hidden, hidden])
            b1 = tf.get_variable('b1_ff', [hidden])
            w2 = tf.get_variable('w2_ff', [hidden, 3])
            b2 = tf.get_variable('b2_ff', [3])
        
        ret_tensor = tf.matmul(inputs, w1) + tf.tile(tf.expand_dims(b1, 0), [batch, 1])
        ret_tensor = tf.matmul(ret_tensor, w2) + tf.tile(tf.expand_dims(b2, 0), [batch, 1])
            
        print 'completed...'
        return ret_tensor
        
        
        
        
        
        