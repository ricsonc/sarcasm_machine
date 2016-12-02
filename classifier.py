#!/usr/bin/env python2

import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
import numpy as np
import random
from keras.regularizers import *

execfile('data_test.py')

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

def test2():
    model = Sequential()
    model.add(GaussianDropout(0.02, input_shape = (l, d)))
    model.add(Bidirectional(GRU(10, return_sequences = True, consume_less='gpu'),
                            merge_mode = 'concat'))
    model.add(Dropout(0.1))
    model.add(GRU(1, consume_less = 'gpu', return_sequences = True))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def test3():
    model = Sequential()
    model.add(Flatten(input_shape=(l,d)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def test4(depth = 2): #deep BD GRU
    condenser = TimeDistributed(Dense(20, activation = 'relu'))
    def bd_layer(prev):
        fwd = GRU(20, return_sequences = True, consume_less = 'gpu',
                  dropout_W = 0.2, dropout_U = 0.2)(prev)
        bck = GRU(20, return_sequences = True, consume_less = 'gpu', go_backwards = True,
                  dropout_W = 0.2, dropout_U = 0.2)(prev)
        return condenser(merge([fwd, bck], mode = 'concat'))
    inputs = Input(shape=(l,d))
    next = inputs
    for i in range(depth):
        next = bd_layer(next)
    summary = GRU(20, consume_less = 'gpu')(next)
    pred = Dense(1, activation = 'sigmoid')(summary)
    return Model(input = inputs, output = pred)


RMS = RMSprop(lr = 0.001)

model = test2()
model.compile(optimizer = RMS,
              loss='binary_crossentropy',
              metrics=['binary_accuracy', 'fbeta_score'])

model.fit(Xtr, ytr, nb_epoch=1000, batch_size=1024, validation_data = (Xte, yte))
