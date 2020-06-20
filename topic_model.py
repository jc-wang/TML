#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np 
class Model(object):

    def __init__(self, vocab_size, embed_dim, h_dim, embeddings, training=True, optimizer='adam'):
        
        self.embeddings = embeddings
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.h_dim = h_dim

        self.bow_repre = tf.placeholder(tf.float32, [None, vocab_size])
        self.customer_bow_repre = tf.placeholder(tf.float32, [None, vocab_size])
        self.agent_bow_repre = tf.placeholder(tf.float32, [None, vocab_size])

        self.id_mask = tf.cast(tf.greater(self.bow_repre, 0), tf.float32)
        self.batch_size = tf.shape(self.bow_repre)[0]

        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.l2_constant = tf.placeholder(tf.float32, [], 'l2_constant')
        self.clip_value = tf.placeholder(tf.float32, [], 'clip_norm')
        self.dropout_keep = tf.placeholder(tf.float32, [], 'dropout')
        self.embeddings = tf.get_variable('topic_model_word_vector', shape=[self.vocab_size, 100], dtype=tf.float32, initializer=tf.constant_initializer(self.embeddings))

        loss1 = self.build_topic_model(self.bow_repre, 'dialog_topic_model')
        loss2 = self.build_topic_model(self.customer_bow_repre, 'customer_topic_model')
        loss3 = self.build_topic_model(self.agent_bow_repre, 'agent_topic_model')
    
        
        self.no_l2_loss = loss1 + loss2 + loss3
        l2_partial_sum = sum([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        l2_loss = tf.multiply(self.l2_constant, l2_partial_sum)
        self.loss = tf.add(self.no_l2_loss, l2_loss)
        if training:
            self._create_training_tensors(optimizer)


    def build_topic_model(self, bow_repre, scope):
        with tf.variable_scope(scope):
            id_mask = tf.cast(tf.greater(bow_repre, 0), tf.float32)
            with tf.variable_scope("encoder"):
                l1_lin = tf.layers.dense(bow_repre, 2*self.embed_dim, activation=tf.nn.tanh, use_bias=True, name='l1')
                mu = tf.layers.dense(l1_lin, self.embed_dim, use_bias=True, name='mu')
                log_sigma_sq = tf.layers.dense(l1_lin, self.embed_dim, use_bias=True, name='log_sigma_sq')

                eps = tf.random_normal((self.batch_size, self.embed_dim), 0, 1, dtype=tf.float32)
                sigma = tf.sqrt(tf.exp(log_sigma_sq))

                h = tf.add(mu, tf.multiply(sigma, eps))

                theta = tf.nn.softmax(tf.layers.dense(h, self.h_dim, name='transform'))
                e_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), 1)
                e_loss = tf.reduce_mean(e_loss)

            with tf.variable_scope("generator"):
                
                t = tf.get_variable('topicVector', shape=[self.h_dim, 100], dtype=tf.float32)
                beta = tf.nn.softmax(tf.matmul(t, self.embeddings, transpose_b=True))

                logits = tf.log(tf.matmul(theta, beta))
                g_loss = -tf.reduce_sum(tf.multiply(logits, id_mask), 1)
                g_loss = tf.reduce_mean(g_loss)
            return g_loss + e_loss

    def _create_training_tensors(self, optimizer_algorithm):
        with tf.name_scope('training'):
            global_step = tf.Variable(0, trainable=False)
            if optimizer_algorithm == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.9)
            elif optimizer_algorithm == 'sgd':
                learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 500, 0.9, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer_algorithm == 'adam':
                optimizer = tf.train.AdamOptimizer()
            else:
                ValueError('Unknown optimizer: %s' % optimizer_algorithm)
            
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)   




# topic model