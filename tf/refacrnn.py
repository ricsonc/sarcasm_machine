#!/usr/bin/env python2

import tensorflow as tf
import random
import numpy as np
from tensorflow.python.ops import functional_ops

def linear(x):
    W = tf.get_variable(name = 'W', initializer=tf.constant_initializer(1.0))
    b = tf.get_variable(name = 'b', initializer=tf.constant_initializer(0.0))
    return W*x+b

def run_step(x, h):
    h_ = tf.nn.tanh(linear(x) + linear(h))
    o = tf.nn.tanh(linear(h_))
    return (h_, o)
