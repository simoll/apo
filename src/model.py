import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.multiply(a, b, name="c")

    sess.run(tf.global_variables_initializer())

    print(a.eval()) # 5.0
    print(b.eval()) # 6.0
    print(c.eval()) # 30.0

    tf.train.write_graph(sess.graph_def, 'build/', 'apo_graph.pb', as_text=False)

## advanced setup ##
## input sequence of statements:
# (oc : one-hot, opOne, opTwo)
# (oc : one-hot, constant)

## output:
# pointer (ptr network output, one element)
# rule distribution (softmax over finite rule set)

# units:
# LSTM


## basic setup ##
# multi-layer CNN (one-hot encodings everywhere)
