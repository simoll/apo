import tensorflow as tf
import numpy as np

def data_type():
    return tf.float32


# learning rate
learning_rate = 0.001

# number of scalar cells in the LSTM
lstm_size = 256

# matrix size in opcode encoding
oc_dict_size = 16




### OpCode model ###
# op code encoding
num_OpCodes = 9
oc_Ret = 3
oc_Add = 4
oc_Sub = 5


# enable debug output
Debug = False

# set to true for pseudo inputs
DummyRun = True

# number of re-write rules
num_Rules = 17

Training = True

if DummyRun:
    batch_size = 4

    # maximal number of parameters
    num_Params = 5

    # maximal program length
    max_Time = 3

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
    # training batch size
    batch_size = 256

    # maximal program len
    max_Time = 8

    # maximal number of parameters
    num_Params = 5

# input feed

# number of instructions in the program
length_data = tf.placeholder(tf.int32, [batch_size], name="length_data")

# opCode per instruction
oc_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="oc_data")

# first operand index per instruction
firstOp_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="firstOp_data")

# second operand index per instruction
sndOp_data = tf.placeholder(tf.int32, [batch_size, max_Time], name="sndOp_data")

rule_in = tf.placeholder(tf.int32, [batch_size], name="rule_in")

# valid operand index range for this program
lowestOperand=-2
highestOperand=1

# most basic version -> operate over a chain of op codes (just for testing)
with tf.Session() as sess:
    ### OK
    # if DummyRun:
    #     sess.run(tf.global_variables_initializer())
    #     print(oc_data.eval())
    #     print(firstOp_data.eval())
    #     print(sndOp_data.eval())

    # opCode embedding
    with tf.device("/cpu:0"):
        oc_embedding = tf.get_variable("oc_embed", [num_OpCodes, lstm_size], dtype=data_type())
        oc_inputs = tf.nn.embedding_lookup(oc_embedding, oc_data) # [batch_size x idx x lstm_size]

    print("oc_inputs: {}".format(oc_inputs.get_shape())) # [ batch_size x max_len x lstm_size ]

    # one-hot operand encoding
    # firstOp_inputs = tf.one_hot(firstOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())
    # sndOp_inputs = tf.one_hot(smdOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())

    # parameter input
    param_data = tf.get_variable("param_embed", [num_Params, lstm_size])

    # build the network
    zero_batch = tf.zeros([batch_size, lstm_size], dtype=data_type())
    # param_batch = tf.split(tf.tile(param_data, [1, batch_size]), 1, batch_size)
    # print(param_data.get_shape()) # [numParams x lstm_size]
    # print(param_batch.get_shape()) # [batch_size x numParams x lstm_size]

    param_batch = tf.concat([[param_data] * batch_size], 1, name="concat_params")

    if Debug:
        print("param_batch: {}".format(param_batch.get_shape()))
 
    # attach neutral and param matrices
    outputs = [zero_batch]
    for i in range(num_Params):
        outputs.append(param_batch[:, i, :])

    # outputs = [zero_batch] + param_data # [batch_size x time x lstm_size]
    if Debug:
        print(outputs)

    # Cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.BasicRNNCell(lstm_size)

    initial_state = cell.zero_state(dtype=data_type(), batch_size=batch_size)
    
    UseRDN=False
    if UseRDN:   # Recursive Dag Network
        # TODO document
        state = initial_state
        with tf.variable_scope("DAG"): 
          for time_step in range(max_Time):
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("time_{}".format(time_step)): # Recursive Dag Network
                # fetch current inputs
                with tf.variable_scope("inputs"):
                    with tf.variable_scope("opCode"):
                        op_code = oc_inputs[:, time_step, :] # [batch_size x lstm_size]
                        flat_oc = tf.reshape(op_code, [batch_size, -1])

                    with tf.variable_scope("firstOp"):
                        first_tensor = tf.gather(outputs, firstOp_data[:, time_step])#[:, time_step, :] 
                        flat_first = tf.reshape(first_tensor, [batch_size, -1])

                    with tf.variable_scope("sndOp"):
                        snd_tensor = tf.gather(outputs, sndOp_data[:, time_step])#[:, time_step, :]
                        flat_snd = tf.reshape(snd_tensor, [batch_size, -1])

                    # merge into joined input (TODO do we need a compression layer??)
                    time_input = tf.concat([flat_oc, flat_first, flat_snd], axis=1,name="seq_input")

                if Debug:
                    print("op_code: {}".format(op_code.get_shape()))
                    print("first_tensor: {}".format(first_tensor.get_shape()))
                    print("snd_tensor: {}".format(snd_tensor.get_shape()))

                if Debug:
                    print("flat_oce: {}".format(flat_oc.get_shape()))
                    print("flat_first: {}".format(flat_first.get_shape()))
                    print("flat_snd: {}".format(flat_snd.get_shape()))

                if Debug:
                    print("time_inp: {}".format(time_input.get_shape()))

                # invoke cell
                (cell_output, state) = cell(time_input, state)

                outputs.append(cell_output)

          # merge all outputs into a single tensor
          # output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size], name="output")

            rdn_output = cell_output
    else:
        # use a plain LSTM
        inputs = tf.unstack(oc_inputs, num=max_Time, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=initial_state, sequence_length=length_data)
        rdn_output = outputs[-1]

    ### Prediction ###
    logits = tf.layers.dense(inputs=rdn_output, units=num_Rules)

    if Training:
      ref_rule = tf.one_hot(rule_in, axis=-1, depth=num_Rules)
      batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ref_rule, logits=logits, dim=-1)
      loss = tf.reduce_sum(batch_loss, name="cost")
      tf.summary.scalar('loss', loss)

      # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # seems to perform better on the "count-oc_Add-task"

      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step(),
          name="train")

    else:
      pred_rule = tf.nn.softmax(logits, name="rule_out")

    # if DummyRun:
    #     raise SystemExit

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("build/tf_logs", sess.graph)

    tf.global_variables_initializer().run()

    if not DummyRun:
        # init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
        tf.train.write_graph(sess.graph, 'build/', 'apo_graph.pb', as_text=False)
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

    if DummyRun:
        feed_dict()
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

    if DummyRun:
        sess.run(tf.global_variables_initializer())
    # print(oc_inputs.eval())
