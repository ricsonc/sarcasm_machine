#!/usr/bin/env python2

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import expon, geom

execfile('load_data.py')

#print type(Xtr[0].flatten())
Xtr = np.array(map(lambda x:x.flatten(), Xtr))
Xte = np.array(map(lambda x:x.flatten(), Xte))

'''
P = PCA(n_components = 100)
P.fit(Xtr)
Xtr = P.transform(Xtr)
Xte = P.transform(Xte)
'''

#hyperparameterization tuning loop
#note that this does not do any better ...
P = PCA()
L = LogisticRegression()
pipe = Pipeline(steps=[('pca',P),('log',L)])
parameters = {'log__penalty': ['l1','l2'],
              'log__C' : expon(loc=0.0,scale=0.05),
              'pca__n_components' : geom(0.01,loc=0.0),
              'pca__whiten' : [True,False]}
H = RandomizedSearchCV(pipe, parameters, n_iter = 50, verbose = 10)

H.fit(Xtr, ytr)
print H.score(Xte,yte)
print H.score(Xtr,ytr)


'''
print 'logistic'
L = LogisticRegression()
L.fit(Xtr, ytr)
print L.score(Xte,yte)
print L.score(Xtr,ytr)

print 'svm'
S = SVC()
S.fit(Xtr,ytr)
print S.score(Xte,yte)
print S.score(Xtr,ytr)

print 'forest'
R = RandomForestClassifier()
R.fit(Xtr,ytr)
print R.score(Xte,yte)
print R.score(Xtr,ytr)

print 'knn'
K = KNeighborsClassifier(9)
K.fit(Xtr,ytr)
print K.score(Xte,yte)
print K.score(Xtr,ytr)
'''
