#!/usr/bin/env python2

import tensorflow as tf
from tensorflow import variable_scope as scope
import random
import numpy as np
from tensorflow.python.ops import functional_ops
from pprint import pprint

const = tf.constant_initializer
    
def linear(x):
    '''computes a linear function of x'''
    W = tf.get_variable(name = 'W', shape = (), initializer=const(1.0))
    b = tf.get_variable(name = 'b', shape = (), initializer=const(0.0))
    return W*x+b

def gendata(n):
    '''generates random data for testing purposes'''
    xs = []
    ys = []
    for i in range(n):
        rolls = random.randint(10, 20)
        x = [float(random.randint(0,1)) for j in range(rolls)]
        y = float(sum(x) < rolls/2)
        xs.append(x)
        ys.append(y)
    return xs, ys

def initialize_variables():
    '''.....'''
    with scope("loss"): 
        with scope("internal_states"):
            with scope("internal_weights"):
                F = linear(0)
        with scope("output_weights"):
            F = linear(0)
        tf.get_variable("output", shape = ())
        tf.get_variable("ce", shape = ())
        tf.get_variable("accuracy", shape = ())

def compute_loss(predictor):
    '''predictor is a function which takes an input 
       and produces an output tensor'''

    '''
    def loss(inputs, label):
        with scope("loss", reuse = True):
            #generates new node every time this is called
            output = tf.get_variable('output', initializer = const(predictor(inputs)))
            init_ce = const(-(label*clipped_log(output) +
                              (1.0-output)*clipped_log(1.0-label)))
            cross_entropy = tf.get_variable('ce', initializer = init_ce)
            return cross_entropy 
    '''
    def loss(inputs, label):
        with scope("loss", reuse = True):
            #generates new node every time this is called
            output = tf.get_variable('output', initializer = const(predictor(inputs)))
            return tf.square(label-output)
    
    return loss

def _except_(msg):
    raise Exception(msg)

def compute_accuracy(predictor):
    def accuracy(inputs, label):
        with scope("loss", reuse = True):
            #generates new node every time this is called
            output = tf.get_variable('output', initializer = const(predictor(inputs)))
            init_accuracy = tf.cast(tf.equal(tf.round(output), tf.round(label)), tf.float32)
            accuracy = tf.get_variable('accuracy', initializer = const(init_accuracy))
            return accuracy
    return accuracy
       
def batcher(predictor, inputses, labels, fn_generator):
    '''computes loss in batch'''
    inputses_lst = tf.unpack(inputses)
    labels_lst = tf.unpack(labels)
    fn = fn_generator(predictor)
    fn_lst = map(lambda (inputs, label) : fn(inputs, label),
                   zip(inputses_lst, labels_lst))
    return tf.reduce_mean(tf.pack(fn_lst), reduction_indices = [0])
    
def clipped_log(x):
    '''log function with clipping to prevent numerical issues'''
    return tf.log(tf.clip_by_value(x, 1E-10, 1E10))

def dynamic_pad(sequences):
    '''dynamically pads sequences to the maximum length sequence'''
    n = max(map(len, sequences))
    return [seq+[-1 for x in range(n-len(seq))] for seq in sequences]

def RNN_model(inputs):
    '''given some placeholder inputs, returns the output of an rnn'''
    print 'made new rnn model'
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
            Y = tf.reduce_max(H, name = 'Y') #be careful with reduction indices
            return tf.nn.sigmoid(Y)

    return conglomerator(roll_rnn(inputs))

class Trainer(object):

    def __init__(self, model, (Xdata, ydata), batch_size):

        initialize_variables()
        self.data = (Xdata, ydata)
        self.split_data()
        self.model = model
        self.bs = batch_size
        
        self.inputses = tf.placeholder(tf.float32, shape = [self.bs, None], name = 'inputses') 
        self.labels = tf.placeholder(tf.float32, shape = [self.bs], name = 'labels')
        self.loss = batcher(self.model, self.inputses, self.labels, compute_loss)
        self.acc = batcher(self.model, self.inputses, self.labels, compute_accuracy)
        self.train_step = tf.train.AdadeltaOptimizer(0.1, rho = 0.99).minimize(self.loss)

        self.train()
        
    def split_data(self, p_train = 0.4, p_val = 0.1, p_test = 0.5):
        '''splits data into train, validation, and test set'''
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
        
    def draw_sample(self, from_ = 'train'): 
        '''draw some samples from either the train set, validation set, or test set'''
        if from_ == 'train':
            dataset = self.train_data
        elif from_ == 'val':
            dataset = self.val_data
        elif from_ == 'test':
            dataset = self.test_data
        return zip(*[random.choice(dataset) for j in range(self.bs)])
        
    def train(self, numiters = 10000000, verbosity = 100):
        '''train the model'''
        tf_vars = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(tf_vars)
        for i in range(numiters):
            self.run_train_step(self.draw_sample())
            if not (i % verbosity):
                print "training step", i
                thing = self.draw_sample('train')
                print 'the cross entropy is :', self.get_loss(thing)
                #print 'the accuracy is :', self.get_acc(thing)
                #self.print_everything(thing)
    def build_feed(self, (Xs, ys)): 
        '''build the feed dict for the model'''
        return {self.inputses: dynamic_pad(list(Xs)), self.labels : list(ys)}
    def run_train_step(self, samples):
        '''running the train step once given some samples'''
        self.session.run(self.train_step, feed_dict = self.build_feed(samples))
    def get_loss(self, samples):
        '''printing the loss of some data'''
        return self.session.run(self.loss, feed_dict = self.build_feed(samples))
    def get_acc(self, samples):
        '''printing the accuracy of some data'''
        return self.session.run(self.acc, feed_dict = self.build_feed(samples))
    def print_everything(self, samples):
        pprint(samples[0], width = 500)
        print samples[1]
        print self.session.run(self.acc, feed_dict = self.build_feed(samples))

Trainer(RNN_model, gendata(1000000), 100)
