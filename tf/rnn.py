#!/usr/bin/env python2

import tensorflow as tf
from tensorflow import variable_scope as scope
import random
import numpy as np
import random
from tensorflow.python.ops import functional_ops

const = tf.constant_initializer

def clipped_log(x):
    return tf.log(tf.clip_by_value(x, 1E-10, 1E10))
    
def linear(x):
    W = tf.get_variable(name = 'W', shape = (), initializer=const(1.0))
    b = tf.get_variable(name = 'b', shape = (), initializer=const(0.0))
    return W*x+b

class RNN_model(object):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, shape = [None], name = 'inputs')
        self.label = tf.placeholder(tf.float32, shape = (), name = 'label') 
        self.output = self.assemble()
        self.loss = self.compute_loss()
       
    def feed(self, data, labels):
        pass
    
    def rnn_instep(self, x, h):
        with scope("internal_weights"):
            h_ = tf.nn.tanh(linear(x) + linear(h))
        return h_
        
    def roll_rnn(self):
        ### really sketchy
        with scope("internal_states/internal_weights", reuse = None): #just to initialize things
            linear(0)
        ###
        with scope("internal_states", reuse = True):
            states = tf.scan(self.rnn_instep, self.inputs, initializer=1.0, name = 'states')
        with scope("output_weights"):
            outputs = linear(states)
        return outputs
        
    def conglomerator(self, outputs):
        with scope("conglomerate"):
            Y = tf.reduce_max(outputs, name = 'Y') 
            return Y #reduction indices??? #dim

    def assemble(self):
        return self.conglomerator(self.roll_rnn())
       
    def compute_loss(self):
        with tf.variable_scope("loss"):
            cross_entropy = -(self.label*clipped_log(self.output) + 
                              (1.0-self.output)*clipped_log(1.0-self.label))
        return cross_entropy #name?

class Trainer(object):
    def __init__(self, data):
        self.data = data

        datasize = 100000
        self.train_data = zip(*self.gendata(datasize))
        self.cross_data = zip(*self.gendata(datasize))
        self.trial_data = zip(*self.gendata(datasize))
        self.trainer()

    def split_data(self, p_train, p_val, p_test):
        sum_ = p_train + p_val + p_test
        p_train /= sum_
        p_val /= sum_
        p_test /= sum_
        
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
