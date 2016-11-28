#!/usr/bin/env python2

import numpy as np
import random

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

data = list(zip(Xs, ys))
random.shuffle(data)
Xtr, ytr = map(np.array,zip(*data[:1024]))
Xte, yte = map(np.array,zip(*data[1024:]))
