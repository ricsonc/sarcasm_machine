#!/usr/bin/env python2

import keras
from keras.layers import *
from keras.models import Sequential, Model
import numpy as np
import random
from keras.regularizers import *


Xs = np.load('corpus_list.npy')
#n = max(map(len, X))
n = 300
def add_comp(vec):
    return np.append(vec, [1])
def add_dim(sample):
    return map(add_comp, sample)
Xs = map(add_dim, Xs)
def resize(sample):
    if len(sample) > n:
        return sample[:n]
    if len(sample) < n:
        res = np.concatenate((sample, np.zeros((n-len(sample),len(sample[0])))))
        return res

Xs = filter(lambda k : len(k), Xs)
Xs = np.array(map(resize, Xs))
ys = np.load('label_list.npy')
Xs, ys = zip(*filter(lambda (j,k) : j is not None, zip(Xs, ys)))
Xs = np.array(Xs)
ys = np.array(ys)

for i, thing in enumerate(Xs):
    assert np.shape(thing) == (300, 27)

n, l, d = np.shape(Xs)

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
    model.add(Bidirectional(LSTM(10, return_sequences = True, consume_less='gpu'),
                            input_shape = (l, 27), merge_mode = 'concat'))
    model.add(Dropout(0.1))
    model.add(LSTM(1, consume_less = 'gpu', return_sequences = True))
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

data = zip(Xs, ys)
random.shuffle(data)
Xtr, ytr = map(np.array,zip(*data[:1024]))
Xte, yte = map(np.array,zip(*data[1024:]))

model = test4()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['binary_accuracy', 'fbeta_score'])
    
model.fit(Xtr, ytr, nb_epoch=1000, batch_size=1024, validation_data = (Xte, yte))
