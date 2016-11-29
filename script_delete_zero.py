#!/usr/bin/env python2


import numpy as np
corpus = np.load('corpus_list.npy')
corpus_bigram = np.load('corpus_list_bigram.npy')
corpus_trigram = np.load('corpus_list_trigram.npy')
corpus_pol = np.load('corpus_list_polarity_word.npy')
label = np.load('label_list.npy')
wrong = 0
for i in xrange(len(corpus)):
	if len(corpus[i])==0:
		wrong = i

print len(corpus),len(label)
corpus = np.delete(corpus,wrong,0)
corpus_pol = np.delete(corpus_pol,wrong)
label = np.delete(label,wrong)
print len(corpus),len(label)

for c in corpus:
        if len(c) == 0:
                print '???'

'''
for i in xrange(len(corpus_bigram)):
	if len(corpus[i])==0:
		wrong = i
'''

corpus_bigram = np.delete(corpus_bigram,wrong,0)

'''
for i in xrange(len(corpus_trigram)):
	if len(corpus[i])==0:
		wrong = i
'''

corpus_trigram = np.delete(corpus_trigram,wrong,0)

np.save('corpus_list_fixed',corpus)
np.save('label_list_fixed',label)
np.save('corpus_list_bigram_fixed',corpus_bigram)
np.save('corpus_list_trigram_fixed',corpus_trigram)
np.save('corpus_list_polarity_word_fixed',corpus_pol)
