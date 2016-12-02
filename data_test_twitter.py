#!/usr/bin/env python2

import numpy as np
import random

def add_comp(vec):
    return np.append(vec, [1])

def add_dim(sample):
    return map(add_comp, sample)

def resize(n):
    def do_resize(sample):
        if len(sample) >= n:
            return sample[:n]
        if len(sample) < n:
            return np.concatenate(
                (sample,np.zeros((n-len(sample),len(sample[0]))))
            )
    return do_resize

def stack(Xslist):
    for Xs in Xslist:
        print np.shape(Xs)
    samples = zip(*Xslist)
    samples_ = []
    for sample in samples:
        samples_.append(np.array(map(np.concatenate,zip(*sample))))
    return np.array(samples_)
    
def load_files(Xfile_list, yfile, n):
    Xslist = []
    for file in Xfile_list:
        Xs = np.load(file)
        for x in Xs:
            if not len(x):
                print file
        Xs = map(add_dim, Xs)
        Xs = np.array(map(resize(n), Xs))
        Xslist.append(Xs)
    return (stack(Xslist), np.load(yfile))

Xs, ys = load_files(['twitter_corpus_vecinp_fixed.npy'],
                    'twitter_corpus_label_fixed.npy',
                    30)

print np.shape(Xs)
print np.shape(ys)

data = list(zip(Xs, ys))
random.shuffle(data)
K0 = 400000
K1 = 440000
Xtr, ytr = map(np.array,zip(*data[:K0]))
n, l, d = np.shape(Xtr)
Xv, yv = map(np.array,zip(*data[K0:K1]))
Xte, yte = map(np.array,zip(*data[K1:]))

print sum(ys), float(sum(ys))/len(ys)
del ys
del Xs

print 'loaded'
