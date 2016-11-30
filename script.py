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

np.delete(corpus,wrong,0)
np.delete(corpus_pol,wrong)
np.delete(label,wrong)
print len(corpus),len(label)

for i in xrange(len(corpus_bigram)):
	if len(corpus[i])==0:
		wrong = i

np.delete(corpus_bigram,wrong,0)

for i in xrange(len(corpus_trigram)):
	if len(corpus[i])==0:
		wrong = i

np.delete(corpus_trigram,wrong,0)

np.save('corpus_list',corpus)
np.save('label_list',label)
np.save('corpus_list_bigram',corpus_bigram)
np.save('corpus_list_trigram',corpus_trigram)
np.save('corpus_list_polarity_word',corpus_pol)
