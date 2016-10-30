#!/usr/bin/env python2

import tensorflow as tf
import random
import numpy as np

def nnlayer(in_vec, dim_in, dim_out, func):
    W = tf.Variable(tf.zeros([dim_out, dim_in]))
    b = tf.Variable(tf.ones([dim_out]))
    return func(tf.matmul(W, in_vec) + b)
                    
def vec_lstm(h_in, c_in, x_in, k):
    g = concat(h_in, x_in)
    nnlayer_ = lambda f : nnlayer(h, 2*k, k, f)
    forget = nnlayer_(tf.sigmoid)
    c_inter = c_in * forget #pointwise
    i_in = nnlayer_(tf.sigmoid)
    z = nnlayer_(tf.tanh)
    c_out = c_inter + z * i_in #pointwise
    output = nnlayer_(tf.sigmoid)
    h_out = c_out * output
    return (h_out, c_out)

def lstm_layer(input_layer, k):
    h = tf.ones([k])
    c = tf.ones([k])
    output_layer = []
    for input_ in input_layer:
        (h, c) = vec_lstm(h, c, input_, k)
        output_layer.append(h)
    return ouput_layer

def reverse_layer(layer):
    return layer.reverse()

def bdlstm(depth, inputs, k):
    layer = lstm_layer(inputs, k)
    for i in range(depth): #off by 1
        layer = lstm_layer(reverse_layer(layer), k)
    return layer

def layer_wrapper(inputs, k):
    output = tf.zeros([k])
    attentions = lstm_layer(inputs, k)
    for i_vec, attention in zip(inputs, attentions):
        output += attention * input_vec #pointwise
    return nnlayer(output, k, 1, tf.sigmoid) #logistic

#compute cross entropy
#batcher
