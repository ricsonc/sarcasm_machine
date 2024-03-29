#!/usr/bin/env python2

import numpy as np
import random

def add_comp(vec):
    return np.append(vec, [1])
def add_dim(sample):
    return map(add_comp, sample)
def resize(sample, n=20):
    print sample
    if len(sample) > n:
        return sample[:n]
    if len(sample) < n:
        res = np.concatenate(
            (sample,np.zeros((n-len(sample),len(sample[0]))))
        )
        return res

def load_files(Xfile_list, yfile, n = 20):
    X__ = None
    for file in Xfile_list:
        Xs = np.load(file)
        Xs = map(add_dim, Xs)
        Xs = np.array(map(resize, Xs))
        if X__ is None:
            X__ = Xs
        else:
            X__ = np.stack((X__, Xs))
    return (np.array(Xs), np.load(yfile))

load_files(['corpus_list.npy',
            'corpus_list_bigram.npy',
            'corpus_list_trigram.npy',
            'corpus_list_polarity.npy'],
           'label_list.npy',
           300)
