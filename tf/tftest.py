#!/usr/bin/env python2
import tensorflow as tf

def plus2_bad(x):
    s1 = tf.Variable(1.0)
    print s1.name
    p1 = x + s1
    s2 = tf.Variable(1.0)
    print s2.name    
    p2 = p1 + s2
    return p2

def plus1_a(x):
    s = tf.Variable(1.0, name = 's')
    print s.name
    return x + s

def plus1_b(x):
    s = tf.get_variable(name = 's', initializer = 1.0)
    print s.name
    return x + s

#without variable scopes
def plus2_a(x):
    w1 = plus1_a(x)
    y1 = plus1_a(w1)
    return y1

#with variable scopes
def plus2_b(x):
    with tf.variable_scope("first"):
        w2 = plus1_b(x)
    with tf.variable_scope("second"):
        y2 = plus1_b(w2)
    return y2

def test():
    zero = tf.Variable(0.0)
    one = tf.Variable(1.0)
    
    print '\nbad implementation'
    p0 = plus2_bad(zero)
    q0 = plus2_bad(one)
    
    print '\nno scope'
    p1 = plus2_a(zero)
    q1 = plus2_a(one)
        
    print '\nwith scope'
    with tf.variable_scope("addition") as scope:
        p2 = plus2_b(zero)
        scope.reuse_variables()
        q2 = plus2_b(one)
        
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

test()
