#!/usr/bin/env python2

import time

import glob
import keras
from keras.layers import *
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import *
import numpy as np
import random
from keras.regularizers import *

#execfile('data_test.py')
execfile('data_test_twitter.py')

def test0():
    model = Sequential()
    model.add(Flatten(input_shape=(l,d)))
    model.add(Dense(1, activation='sigmoid', W_regularizer = l2(0.2), b_regularizer = l2(0.2)))
    return model

def test1():
    model = Sequential()
    model.add(LSTM(10, input_dim = d, input_length = l,
                   return_sequences = True, consume_less='gpu'))
    model.add(Dropout(0.3))
    model.add(LSTM(10, return_sequences = True, consume_less = 'gpu'))
    model.add(Dropout(0.3))
    model.add(LSTM(1, return_sequences = True, consume_less = 'gpu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def test2(drop = 0.0):
    model = Sequential()
    model.add(GaussianDropout(drop, input_shape = (l, d)))
    model.add(Bidirectional(GRU(10, return_sequences = True, consume_less='gpu'),
                            merge_mode = 'concat'))
    model.add(Dropout(drop))
    model.add(GRU(1, consume_less = 'gpu', return_sequences = True))
    model.add(Flatten())
    model.add(Dropout(drop))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def test4(depth = 3, dim = 10, drop = 0.01): #deep BD GRU
    condenser = TimeDistributed(Dense(dim, activation = 'relu'))
    def bd_layer(prev):
        fwd = GRU(dim, return_sequences = True, consume_less = 'gpu',
                  dropout_W = drop, dropout_U = drop)(prev)
        bck = GRU(dim, return_sequences = True, consume_less = 'gpu', go_backwards = True,
                  dropout_W = drop, dropout_U = drop)(prev)
        return condenser(merge([fwd, bck], mode = 'concat'))
    inputs = Input(shape=(l,d))
    next = inputs
    for i in range(depth):
        next = bd_layer(next)
    summary = GRU(dim, consume_less = 'gpu')(next)
    pred = Dense(1, activation = 'sigmoid')(summary)
    return Model(input = inputs, output = pred)

def test5(sizes = [16, 8, 1], drop = 0.00): #deep BD GRU
    assert sizes[-1] == 1
    def bd_layer(prev, dim):
        fwd = GRU(dim, return_sequences = True, consume_less = 'gpu',
                  dropout_W = drop, dropout_U = drop)(prev)
        bck = GRU(dim, return_sequences = True, consume_less = 'gpu', go_backwards = True,
                  dropout_W = drop, dropout_U = drop)(prev)
        return Dropout(drop)(merge([fwd, bck], mode = 'sum'))

    inputs = Input(shape=(l,d))
    next = inputs

    for i, size in enumerate(sizes):
        next = bd_layer(next, size)

    summary = Dense(10, activation = 'relu')(Flatten()(next))
    pred = Dense(1, activation = 'sigmoid')(summary)
    return Model(input = inputs, output = pred)

def large():
    return test5([16, 8, 4, 8, 4, 1])

def test6(): #convnet
    model = Sequential()
    model.add(Dropout(0.4, input_shape =(l,d)))    
    model.add(Convolution1D(16, 3))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(2))
    model.add(Convolution1D(4, 3))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))    
    model.add(Convolution1D(4, 3))
    model.add(PReLU())
    model.add(MaxPooling1D(2))
    model.add(TimeDistributed(Dense(1, activation = 'relu')))
    model.add(Flatten())
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def test3():
    model = Sequential()
    model.add(Flatten(input_shape=(l,d)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.0))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def test7(): #big ff
    model = Sequential()
    model.add(Flatten(input_shape=(l,d)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))    
    for i in range(20):
        model.add(Highway(activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def test8(): #shallow
    model = Sequential()
    model.add(Flatten(input_shape=(l,d)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

OPT = Nadam()

modelname = "s3.h5"

if modelname in glob.glob('*.h5'):
    model = load_model(modelname)
else:
    model = test8()
    model.compile(optimizer = OPT,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'fbeta_score'])
    #model.save(modelname)

model.compile(optimizer = OPT,
              loss='binary_crossentropy',
              metrics=['binary_accuracy', 'fbeta_score'])

ne = 1000000
while 1:
    model.fit(Xtr, ytr, nb_epoch=ne, batch_size=1000, validation_data = (Xv, yv))
    model.save(modelname)
    model.save(str(int(time.time()))+modelname)

print model.evaluate(Xte,yte)
    
