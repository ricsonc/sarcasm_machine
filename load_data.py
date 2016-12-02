#!/usr/bin/env python2

import numpy as np
import random

execfile('data_test.py')

n, l, d = np.shape(Xs)

data = list(zip(Xs, ys))
random.shuffle(data)
Xtr, ytr = map(np.array,zip(*data[:1024]))
Xte, yte = map(np.array,zip(*data[1024:]))
