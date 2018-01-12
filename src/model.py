import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

def data_type():
    return tf.float32




### OpCode model ###
# op code encoding
num_OpCodes = 10
oc_Ret = 3
oc_Add = 4
oc_Sub = 5


# enable debug output
Debug = False

# set to true for pseudo inputs
DummyRun = False

Training = True

def parseConfig(fileName):
  res = dict()
  for line in open(fileName, 'r'):
    parts = line.split(" ")
    res[parts[0]] = parts[1]
  return res

if DummyRun:
    batch_size = 4

    # maximal number of parameters
    num_Params = 5

    # maximal program length
    prog_length = 3

    # number of re-write rules
    num_Rules = 17

    # number of scalar cells in the LSTM
    state_size = 64
    
    # op code embedding size
    embed_size = 32
    
    # stacked cells
    num_layers = 2

    # some test data
    # 0 == dummy
    # 1 == param 1
    # 2 == param 2
    # 3 == param 3
    # 4 == instruction @ 0

    # program = tf.constant([[[oc_Add, 1, 2], [oc_Sub, 4, 1], [oc_Ret, 5, 0]]])
    # oc_data = tf.reshape(tf.slice(program, [0, 0, 0], [-1, -1, 1]), [batch_size, -1])
    # print("oc_data: {}".format(oc_data.get_shape())) # [batch_size x max_len]
    # 
    # firstOp_data = tf.reshape(tf.slice(program, [0, 0, 1], [-1, -1, 1]), [batch_size, -1])
    # print("firstOp_data: {}".format(firstOp_data.get_shape())) # [batch_size x max_len]
    # 
    # sndOp_data = tf.reshape(tf.slice(program, [0, 0, 2], [-1, -1, 1]), [batch_size, -1])

    # return the number of instructions
    # rule_in = tf.constant([3])
else:
    conf = parseConfig("model.conf")

    # maximal program len
    prog_length = int(conf["prog_length"])

    # maximal number of parameters
    num_Params = int(conf["num_Params"])

    # number of re-write rules
    num_Rules = int(conf["num_Rules"]) #, 17

    # number of scalar cells in the LSTM
    state_size = int(conf["state_size"]) #64
    
    # op code embedding size
    embed_size = int(conf["embed_size"]) #32
    
    # stacked cells
    num_layers = int(conf["num_layers"]) #2

    # cell_type
    cell_type= conf["cell_type"]
    print("Model (construct). prog_length={}, num_Params={}, num_Rules={}, embed_size={}, num_layers={}, cell_type={}".format(prog_length, num_Params, num_Rules, embed_size, num_layers, cell_type))
# input feed

# training control parameter (has auto increment)
global_step = tf.get_variable("global_step", initializer = 0, dtype=tf.int32, trainable=False)

# number of instructions in the program
length_data = tf.placeholder(tf.int32, [None], name="length_data")
batch_size=tf.shape(length_data)[0]

# opCode per instruction
oc_data = tf.placeholder(tf.int32, [None, prog_length], name="oc_data")

# first operand index per instruction
firstOp_data = tf.placeholder(tf.int32, [None, prog_length], name="firstOp_data")

# second operand index per instruction
sndOp_data = tf.placeholder(tf.int32, [None, prog_length], name="sndOp_data")





with tf.Session() as sess:
    ### OK
    # if DummyRun:
    #     sess.run(tf.global_variables_initializer())
    #     print(oc_data.eval())
    #     print(firstOp_data.eval())
    #     print(sndOp_data.eval())

    # opCode embedding
    with tf.device("/cpu:0"):
        oc_embedding = tf.get_variable("oc_embed", [num_OpCodes, embed_size], dtype=data_type())
        oc_inputs = tf.nn.embedding_lookup(oc_embedding, oc_data) # [batch_size x idx x embed_size]

    print("oc_inputs: {}".format(oc_inputs.get_shape())) # [ batch_size x max_len x embed_size ]

    # one-hot operand encoding
    # firstOp_inputs = tf.one_hot(firstOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())
    # sndOp_inputs = tf.one_hot(smdOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())

    # parameter input
    param_data = tf.get_variable("param_embed", [num_Params, state_size])

    # build the network
    zero_batch = tf.zeros([batch_size, state_size], dtype=data_type())
    # param_batch = tf.split(tf.tile(param_data, [1, batch_size]), 1, batch_size)
    # print(param_data.get_shape()) # [numParams x state_size]
    # print(param_batch.get_shape()) # [batch_size x numParams x state_size]

    param_batch = tf.reshape(tf.tile(param_data, [batch_size, 1]), [batch_size, num_Params, -1])

    if Debug:
        print("param_batch: {}".format(param_batch.get_shape()))
 
    # attach neutral and param matrices
    outputs = [zero_batch]
    for i in range(num_Params):
        outputs.append(param_batch[:, i, :])

    # outputs = [zero_batch] + param_data # [batch_size x time x state_size]
    if Debug:
        print(outputs)

    ### recurrent cell setup ###
    tupleState=False
    if cell_type== "gru":
        make_cell = lambda: tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == "block":
        tupleState = True
        make_cell = lambda: tf.contrib.rnn.LSTMBlockCell(state_size)
    else:
        tupleState = True
        make_cell = lambda: tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
      # cell = tf.nn.rnn_cell.BasicRNNCell(state_size)


    initial_outputs = outputs
    
    ### network setup ###
    UseRDN=True 
    if UseRDN:   # Recursive Dag Network
        # TODO document
        with tf.variable_scope("DAG"): 
          out_states=[]
          for l in range(num_layers):
            if l == 0:
              # apply LSTM to opCodes
              inputs = tf.unstack(oc_inputs, num=prog_length, axis=1)
            else:
              # DEBUG
              # inputs = outputs
              # pass

              # last iteration output states
              sequence = [tf.zeros([batch_size, state_size], dtype=data_type())] * (1 + num_Params) + outputs
              # print(sequence)

              # next layer inputs to assemble
              inputs=[]
              batch_range = tf.expand_dims(tf.range(0, batch_size), axis=1) # [batch_size x 1]
              for time_step in range(prog_length):
                # if time_step > 0: tf.get_variable_scope().reuse_variables()

                  # fetch current inputs
                  with tf.variable_scope("inputs"):
                    # gather first operand outputs
                    with tf.variable_scope("firstOp"):
                      # sequence [prog_len x batch_size x state_size]
                      # indices [batch_size x 1]
                      indices = tf.expand_dims(firstOp_data[:, time_step], axis=1)
                      idx = tf.concat([indices, batch_range], axis=1)
                      flat_first = tf.gather_nd(sequence, idx)

                    # gather second operand outputs
                    with tf.variable_scope("sndOp"):
                      indices = tf.expand_dims(sndOp_data[:, time_step], axis=1)
                      idx = tf.concat([indices, batch_range], axis=1)
                      flat_snd = tf.gather_nd(sequence, idx)

                    # merge into joined input 
                    time_input = tf.concat([outputs[time_step], flat_first, flat_snd], axis=1, name="seq_input")
                    inputs.append(time_input)


            with tf.variable_scope("layer_{}".format(l)): # Recursive Dag Network
              # print("Input at layer {}".format(l))
              # print(inputs)
              cell = make_cell()
              initial_state = cell.zero_state(dtype=data_type(), batch_size=batch_size)
              outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=initial_state, sequence_length=length_data)
              if tupleState:
                # e.g. LSTM
                out_states.append(state[0])
                out_states.append(state[1])
              else:
                # e.g. GRU
                out_states.append(state)

            # last_output_size = tf.dim_size(outputs[0], 0)

        # last layer output state
        net_out = tf.concat(out_states, axis=1)

    else:
        # multi layer cell
        if num_layers > 1:
          cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)], state_is_tuple=True)
        else:
          cell = make_cell()

        initial_state = cell.zero_state(dtype=data_type(), batch_size=batch_size)

        # use a plain LSTM
        inputs = tf.unstack(oc_inputs, num=prog_length, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=initial_state, sequence_length=length_data)# swap_memory=True)
        last_output = outputs[-1]
        if num_layers > 1:
          net_out = tf.reshape(state[-1].c, [batch_size, -1])
        else:
          net_out = state.c

    # fold hidden layer to decision bits
    rule_logits = tf.layers.dense(inputs=net_out, units=num_Rules)
    target_logits = tf.layers.dense(inputs=net_out, units=prog_length) # TODO use ptr-net instead

    ## predictions ##
    # distributions
    tf.nn.softmax(logits=rule_logits,name="pred_rule_dist")
    tf.nn.softmax(logits=target_logits,name="pred_target_dist")

    # most-likely categories
    pred_rule = tf.cast(tf.argmax(rule_logits, axis=1), tf.int32, name="pred_rule")
    pred_target = tf.cast(tf.argmax(target_logits, axis=1), tf.int32, name="pred_target")

    if False:
        ### target pointer extraction (pointer net) ###
        def attention(ref, query, scope="attention"):
          with tf.variable_scope(scope):
            W_ref = tf.get_variable(
                "W_ref", [state_size, state_size])
            W_q = tf.get_variable(
                "W_q", [state_size, state_size])
            v = tf.get_variable(
                "v", [state_size])
        
            encoded_ref = tf.matmul(ref, W_ref, name="encoded_ref")
            encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)
            tiled_encoded_Query = tf.tile(
                encoded_query, [1, tf.shape(encoded_ref)[1], 1], name="tiled_encoded_query")
            scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])
            return scores
        
        def glimpse(ref, query, scope="glimpse"):
          p = tf.nn.softmax(attention(ref, query, scope=scope))
          alignments = tf.expand_dims(p, 2)
          return tf.reduce_sum(alignments * ref, [1])

    ### reference input & training ###
    # reference input #
    rule_in = tf.placeholder(data_type(), [None, num_Rules], name="rule_in")
    target_in = tf.placeholder(data_type(), [None, prog_length], name="target_in")

    # training #
    # ref_rule = tf.one_hot(rule_in, axis=-1, depth=num_Rules)
    # rule_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ref_rule, logits=rule_logits, dim=-1)
    # ref_target = tf.one_hot(target_in, axis=-1, depth=max_Time)
    # target_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ref_target, logits=target_logits, dim=-1)

    rule_loss = tf.nn.softmax_cross_entropy_with_logits(labels=rule_in, logits=rule_logits, dim=-1)
    target_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_in, logits=target_logits, dim=-1)

    mean_rule_loss = tf.reduce_mean(rule_loss, name="mean_rule_loss")
    mean_target_loss = tf.reduce_mean(target_loss, name="mean_target_loss")

    all_losses = [rule_loss, target_loss]
    loss = tf.reduce_mean(all_losses, name="loss")
    tf.summary.scalar('loss', loss)

    # learning rate configuration
    # starter_learning_rate = 0.1
    # end_learning_rate = 0.0001
    # decay_steps = 400000
    # learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
    #                                           decay_steps, end_learning_rate,
    #                                           power=0.5, name="learning_rate")
    
    # learning rate parameter
    learning_rate = tf.get_variable("learning_rate", initializer=0.0001, dtype=tf.float32, trainable=False)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # seems to perform better on the "count-oc_Add-task"

    train_dist_op = optimizer.minimize(
        loss=loss,
        global_step=global_step,
        name="train_dist_op")

    ### prob of getting the cout right (pCorrect_op)  ###

    # def equals_fn(x,y):
    #   return 1 if x == y else 0

    # matched_rule = tf.cast(tf.equal(pred_rule, rule_in), tf.float32) #tf.map_fn(equals_fn, zip(predicted, rule_in), dtype=tf.int32, back_prop=false)
    # matched_target = tf.cast(tf.equal(pred_target, target_in), tf.float32) #tf.map_fn(equals_fn, zip(predicted, rule_in), dtype=tf.int32, back_prop=false)
    # pCorrect = tf.reduce_mean([matched_rule, matched_target], name="pCorrect_op")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("build/tf_logs", sess.graph)

    tf.global_variables_initializer().run()

    if not DummyRun:
        init = tf.variables_initializer(tf.global_variables(), name='init_op')
        fileName = "apo_graph.pb"
        modelPrefix ="build/rdn"

        # save metagraph
        tf.train.Saver(tf.trainable_variables()).save(sess, modelPrefix) 

        # tf.train.write_graph(sess.graph, 'build/', fileName, as_text=False)
        print("Model written to {}.".format(modelPrefix))
        writer.close()
        raise SystemExit

    def feed_dict():
        # oc_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="oc_feed")
        # firstOp_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="firstOp_feed")
        # sndOp_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="sndOp_feed")
        # rule_in = tf.placeholder(tf.int32, [batch_size], name="rule_in")
        program = tf.constant([[[oc_Add, 1, 2], [oc_Sub, 4, 1], [oc_Ret, 5, 0]]])
        # oc_dummy = tf.reshape(tf.slice(program, [0, 0, 0], [-1, -1, 1]), [batch_size, -1])


        # print("oc_dummy: {}".format(oc_dummy.get_shape())) # [batch_size x max_len]
        
        oc_dummy=[[oc_Add, oc_Sub, oc_Ret],
                  [oc_Add, oc_Sub, oc_Add],
                  [oc_Sub, oc_Sub, oc_Sub],
                  [oc_Add, oc_Add, oc_Add]]

        rule_in_dummy = [1, 2, 0, 3] # number of oc_Add s
        firstOp_dummy = [[1, 4, 5]] * batch_size
        sndOp_dummy = [[2, 1, 0]] * batch_size
        length_dummy=[3] * batch_size

        # firstOp_dummy = tf.reshape(tf.slice(program, [0, 0, 1], [-1, -1, 1]), [batch_size, -1])
        # print("firstOp_data: {}".format(firstOp_data.get_shape())) # [batch_size x max_len]
        
        # return the number of instructions
        return {oc_data: oc_dummy, firstOp_data: firstOp_dummy, sndOp_data: sndOp_dummy, rule_in: rule_in_dummy, length_data: length_dummy}

    if merged is None:
        print(" merged was none!!")
        raise SystemExit

    print(merged)
    print(loss)
    print(train_op)

    train_steps=10000
    for i in range(train_steps):
      if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, roundLoss  = sess.run([merged, loss], feed_dict=feed_dict())
        writer.add_summary(summary, i)
        print('Loss at step %s: %s' % (i, roundLoss))
    
      else:  # Record train set summaries, and train
        summary, _ = sess.run([merged, train_op], feed_dict=feed_dict())
        writer.add_summary(summary, i)

    writer.close()

    # return output, state

    # one LSTM invocation
    # for inst in oc_inputs:
    #    output, state = lstm(inst, state)
    # seq_length=[4] * batch_size
    # outputs, state = tf.nn.static_rnn(cell, tf.unstack(oc_data, 1), initial_state=initial_state, sequence_length=seq_length)

    sess.run(tf.global_variables_initializer())
    # print(oc_inputs.eval())
