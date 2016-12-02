#!/usr/bin/env python2

import numpy as np
corpus = np.load('twitter_corpus_vecinp.npy')
label = np.load('twitter_corpus_label.npy')
wrong = []


for i in xrange(len(corpus)):
	if len(corpus[i])==0:
		wrong.append(i)

print len(corpus),len(label)
corpus = np.delete(corpus,wrong,0)
label = np.delete(label,wrong)
print len(corpus),len(label)

for c in corpus:
        if len(c) == 0:
                print '???'

np.save('twitter_corpus_vecinp_fixed',corpus)
np.save('twitter_corpus_label_fixed',label)
