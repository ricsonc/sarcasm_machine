#!/usr/bin/env python2

import tensorflow as tf
from tensorflow import variable_scope as scope
import random
import numpy as np
import random
from tensorflow.python.ops import functional_ops

const = tf.constant_initializer
    
def linear(x):
    W = tf.get_variable(name = 'W', shape = (), initializer=const(1.0))
    b = tf.get_variable(name = 'b', shape = (), initializer=const(0.0))
    return W*x+b


def RNN_model(inputs):

    def rnn_instep(x, h):
        with scope("internal_weights"):
            h_ = tf.nn.tanh(linear(x) + linear(h))
        return h_

    def roll_rnn(inputs):
        ### really sketchy
        with scope("internal_states/internal_weights", reuse = None): #just to initialize things
            linear(0)
        ###
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

def compute_loss(predictor):
    def loss(inputs, label):
        with scope("loss"):
            output = predictor(inputs)
            cross_entropy = -(label*clipped_log(output) + 
                              (1.0-output)*clipped_log(1.0-label))
            return cross_entropy #name?
    return loss

def batch_loss(predictor, inputss, labels):
    return tf.map_fn(compute_loss(predictor), _)

#use tf.map_fn to batch everything
#we need to write things as functions of inputs 
    
def clipped_log(x):
    return tf.log(tf.clip_by_value(x, 1E-10, 1E10))

class Trainer(obect):
    def __init__(self, model, data):
        self.loss = self.compute_loss()
        self.data = data

    def split_data(self, p_train, p_val, p_test):
        sum_ = p_train + p_val + p_test
        p_train /= sum_
        p_val /= sum_
        p_test /= sum_
        
    def batch_loss(self, data):
        inputs, labels = zip(*data)
        
class Trainer(object):
    def __init__(self, data):
        self.data = data

        datasize = 100000
        self.train_data = zip(*self.gendata(datasize))
        self.cross_data = zip(*self.gendata(datasize))
        self.trial_data = zip(*self.gendata(datasize))
        self.trainer()

    def trainer(self):
        M = RNN_model()
        train_step = tf.train.AdadeltaOptimizer().minimize(M.loss)
        tf_vars = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(tf_vars)
        self.train(M, 100000, 100, train_step)

    #def batch_loss(fn, inputs, labels):
    #    pass

    def gendata(self, n):
	xs = []
        ys = []
        for i in range(n):
	    rolls = random.randint(10, 20)
	    x = [random.randint(0,1) for j in range(rolls)]
	    y = int(sum(x) < rolls/2)
	    xs.append(x)
	    ys.append(y)
        return xs, ys

    #def drawsample(data):
    #    x, y = zip(*[random.choice(data) for j in range(batchsize2)])
    #    xt = np.transpose(np.array(x))
    #    return xt, y

    def drawsample(self, data):
        return random.choice(data)

    def train(self, model, train_iters, printn, step):	  
        print 'training'
        for i in range(train_iters):
	    sx, sy= self.drawsample(self.train_data)
	    print sx, sy
	    self.session.run(step, feed_dict={model.inputs:sx, model.label:sy})
	    if not (i % printn) and i:
	        sx, sy = self.drawsample(self.train_data)
	        print self.session.run(model.loss_, feed_dict={model.inputs:sx, model.label:sy}),
	        sx, sy = self.drawsample(self.cross_data)	   
	        print self.session.run(model.loss_, feed_dict={model.inputs:sx, model.label:sy})


Trainer()
