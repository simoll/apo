import tensorflow as tf
import numpy as np



def data_type():
    return tf.float32

with tf.Session() as sess:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.multiply(a, b, name="c")

    sess.run(tf.global_variables_initializer())

    print(a.eval()) # 5.0
    print(b.eval()) # 6.0
    print(c.eval()) # 30.0

    tf.train.write_graph(sess.graph_def, 'build/', 'toy_graph.pb', as_text=False)

# number of distinct opcodes
num_codes = 9

# most basic version -> operate over a chain of op codes (just for testing)
with tf.Session() as sess:
    batch_size = 256
    num_features = 1
    lstm_size = 256

    # op code translation
    oc_data = tf.placeholder(data_type(), [None, batch_size, num_features], name = "oc")

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [num_codes, lstm_size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, oc_data)

    # sess.run(tf.global_variables_initializer())

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

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
