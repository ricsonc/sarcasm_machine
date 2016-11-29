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
    return np.array(map(np.concatenate,(zip(*Xslist))))
    
def load_files(Xfile_list, yfile, n = 20):
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

load_files(['corpus_list_fixed.npy',
            'corpus_list_bigram_fixed.npy',
            'corpus_list_trigram_fixed.npy',
            'corpus_list_polarity_word_fixed.npy'],
           'label_list.npy',
           300)
