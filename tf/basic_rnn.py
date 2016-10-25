#!/usr/bin/env python2

import tensorflow as tf
import random
import numpy as np

datasize = 200000
train_iters = 20000
printn = 1000

rolls = 12
batchsize = None
batchsize2 = 100

#most difficult
def training_data0(n = 0):
    xs = []
    ys = []
    for i in range(n if n else datasize):
        x = [random.randint(0,1) for j in range(rolls)]
        y = int((sum(x) < rolls/3) or (sum(x) > 1*rolls/2))
        xs.append(x)
        ys.append(y)
    return xs, ys

def training_data1(n = 0):
    xs = []
    ys = []
    for i in range(n if n else datasize):
        x = [random.randint(0,1) for j in range(rolls)]
        y = int(sum(x) >= rolls/2)
        xs.append(x)
        ys.append(y)
    return xs, ys

def training_data2(n = 0):
    xs = []
    ys = []
    for i in range(n if n else datasize):
        y = random.randint(0, 1)
        x = [random.uniform(0,1)*(y-0.5) for j in range(rolls)]
        xs.append(x)
        ys.append(y)
    return xs, ys

#easiest
def training_data3(n = 0):
    xs = []
    ys = []
    for i in range(n if n else datasize):
        y = random.randint(0, 1)
        x = [y for j in range(rolls)] 
        xs.append(x)
        ys.append(y)
    return xs, ys

training_data = training_data0

##the function to use
func = tf.nn.sigmoid #relu not working???
trainrate = 10.0 

#the memory
x = tf.Variable(1.0, 'x')
#the inputs
i_ = tf.placeholder(tf.float32, [rolls, batchsize])
#the weights
w1 = tf.Variable(1.0, 'w1')
w2 = tf.Variable(1.0, 'w2')
w3 = tf.Variable(1.0, 'w3')
#the biases
b1 = tf.Variable(0.0, 'b1')
b2 = tf.Variable(0.0, 'b2')
#the outputs
ys = []
for elem in tf.unpack(i_):
    x = func(w1*elem+w2*x + b1)
    ys.append(func(x*w3 + b2))
#averaged output
y = tf.reduce_mean(tf.pack(ys), name = 'avg', reduction_indices = [0])
#correct label
y_ = tf.placeholder(tf.float32, [batchsize])
#cross entropy
softlog = lambda V : tf.log(tf.clip_by_value(V, 1E-10, 1E10))
CE = -tf.reduce_mean(y_*softlog(y) + (1.0-y_)*softlog(1.0-y))
#training
train_step = tf.train.AdadeltaOptimizer(trainrate, rho = 0.9).minimize(CE)
#initialize session
tf_vars = tf.initialize_all_variables()
session = tf.Session()
session.run(tf_vars)

#generate training data
print 'generating data'
traindata = zip(*training_data())
traindata2 = zip(*training_data())
testx, testy = training_data()

def drawsample(data):
    x, y = zip(*[random.choice(data) for j in range(batchsize2)])
    xt = np.transpose(np.array(x))
    return xt, y

print 'training'
for i in range(train_iters):
    samplei, sampley = drawsample(traindata)
    session.run(train_step, feed_dict={i_:samplei, y_:sampley}) 
    if not (i % printn) and i:
        samplei, sampley = drawsample(traindata)
        print session.run(CE, feed_dict={i_:samplei, y_:sampley}),
        samplei, sampley = drawsample(traindata2)        
        print session.run(CE, feed_dict={i_:samplei, y_:sampley})
        
#testing now
print 'testing'
score = tf.reduce_mean(tf.cast(tf.equal(tf.round(y), y_), tf.float32))
print 'accuracy is', session.run(score, feed_dict={i_:np.transpose(np.array(testx)), y_:testy})

def testvars(vars, session):
    xs, ys = training_data(5)
    return session.run([vars], feed_dict ={i_:np.transpose(np.array(xs)), y_:ys})

print testvar([w1, w2, w3, b1, b2, y, y_, CE], session)
