#!/usr/bin/env python2

import numpy as np
import random

def add_comp(vec):
    return np.append(vec, [1])
def add_dim(sample):
    return map(add_comp, sample)
def resize(sample):
    if len(sample) > n:
        return sample[:n]
    if len(sample) < n:
        res = np.concatenate(
            (sample,np.zeros((n-len(sample),len(sample[0]))))
        )
        return res

def load_files(Xfile_list, yfile_list, n = 20):
    X__ = None
    for file in file_list:
        Xs = np.load(file)
        Xs = map(add_dim, Xs)
        Xs = np.array(map(resize, Xs))
        if X__ is None:
            X__ = Xs
        else:
            X__ = np.stack((X__, Xs))
    ys = np.load('label_list.npy')
    return (np.array(Xs), np.array(ys))

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
        res = np.concatenate(
            (sample,np.zeros((n-len(sample),len(sample[0]))))
        )
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
