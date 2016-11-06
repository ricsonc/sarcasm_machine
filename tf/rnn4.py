#!/usr/bin/env python2

import tensorflow as tf
from tensorflow import variable_scope as scope
import random
import numpy as np
from tensorflow.python.ops import functional_ops

const = tf.constant_initializer
    
def linear(x):
    W = tf.get_variable(name = 'W', shape = (), initializer=const(1.0))
    b = tf.get_variable(name = 'b', shape = (), initializer=const(0.0))
    return W*x+b

#inputs expected to be placeholder variables
def RNN_model(inputs):
    
    def rnn_instep(x, h):
        with scope("internal_weights"):
            h_ = tf.nn.tanh(linear(x) + linear(h))
        return h_

    def roll_rnn(inputs):
        with scope("internal_states", reuse = True):
            states = tf.scan(rnn_instep, inputs, initializer=1.0, name = 'states')
        with scope("output_weights"):
            outputs = linear(states)
        return outputs
        
    def conglomerator(H):
        with scope("conglomerate"):
            Y = tf.reduce_max(H, name = 'Y') 
            return Y #reduction indices??? #dim

    return conglomerator(roll_rnn(inputs))

#predictor here is a function that predicts, such as RNN_model
def compute_loss(predictor): 
    def loss(inputs, label):
        with scope("loss", reuse = True):
            output = predictor(inputs) #this generates 100 nodes!
            cross_entropy = -(label*clipped_log(output) + 
                              (1.0-output)*clipped_log(1.0-label))
            return cross_entropy #name?
    return loss

#goal : run the loss on each inputs and label, and then averages all the losses
#1. unpack
#2. python map
#3. pack
def batch_loss(predictor, inputses, labels):
    #unstack here returns a list of columns, each one of which is a time series
    inputses_lst = tf.unpack(inputses)
    #and here it returns a list of scalar labels
    labels_lst = tf.unpack(labels)
    loss_fn = compute_loss(predictor)
    #map the loss function over the list of tuples
    loss_lst = map(lambda (inputs, label) : loss_fn(inputs, label), zip(inputses_lst, labels_lst))
    return tf.reduce_mean(tf.pack(loss_lst))
    
def clipped_log(x):
    return tf.log(tf.clip_by_value(x, 1E-10, 1E10))

#under construction
class Trainer(object):
    def __init__(self, model, data, batch_size): #data = (list of xs, list of ys)
        self.data = data
        self.split_data()
        self.model = model
        self.bs = batch_size
        
        self.inputses = tf.placeholder(tf.float32, shape = [self.bs, None]) #may be wrong axis
        self.labels = tf.placeholder(tf.float32, shape = [self.bs])
        self.loss = batch_loss(self.model, self.inputses, self.labels)
        self.train_step = tf.train.AdadeltaOptimizer().minimize(self.loss)
    
        self.train()
        
    def split_data(self, p_train = 0.4, p_val = 0.1, p_test = 0.5):
        sum_ = p_train + p_val + p_test
        p_train /= sum_
        p_val /= sum_
        p_test /= sum_

        slot = zip(self.data[0], self.data[1])
        random.shuffle(slot)
        numData = len(slot)
        
        # train_data, val_data, test_data are lists of tuples, not tuples of lists
        self.train_data = slot[:int(p_train*numData)]
        self.val_data = slot[int(p_train*numData):int((p_train+p_val)*numData)]
        self.test_data = slot[int((p_train+p_val)*numData):]
        
    def draw_sample(self, from_ = 'train'): #from can be 'train', 'val', or 'test'
        if from_ == 'train':
            dataset = self.train_data
        elif from_ == 'val':
            dataset = self.val_data
        elif from_ == 'test':
            dataset = self.test_data
        return zip(*[random.choice(dataset) for j in range(self.bs)])
        
    def train(self, numiters = 10000, verbosity = 100):
        tf_vars = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(tf_vars)
        for i in range(numiters):
            self.run_train_step(self.draw_sample())
            if not (i % verbosity):
                print "training step", i
                print self.get_loss(self.draw_sample('val'))

    def build_feed(self, (Xs, ys)): #xs has unequal lens
        return {self.inputses: list(Xs), self.labels : list(ys)}

    def run_train_step(self, samples): 
        self.session.run(self.train_step, feed_dict = self.build_feed(samples))

    def get_loss(self, samples):
        self.session.run(self.loss, feed_dict = self.build_feed(samples))
    
def gendata(n):
    xs = []
    ys = []
    for i in range(n):
        rolls = random.randint(10, 20)
        x = [random.randint(0,1) for j in range(rolls)]
        y = int(sum(x) < rolls/2)
        xs.append(x)
        ys.append(y)
    return xs, ys

### really sketchy
with scope("loss"): #just to initialize things
    with scope("internal_states"):
        with scope("internal_weights"):
            F = linear(0)
    with scope("output_weights"):
        F = linear(0)
###

Trainer(RNN_model, gendata(100000), 100)
