import tensorflow as tf
import numpy as np

def data_type():
    return tf.float32

### OpCode model ###
# op code encoding
num_OpCodes = 9
oc_Ret = 3
oc_Add = 4
oc_Sub = 5

# number of scalar cells in the LSTM
lstm_size = 256

# size of opcode embedding
oc_dict_size = 16

Debug = False

# set to true for pseudo inputs
DummyRun = False

if DummyRun:
    batch_size = 1

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

    program = tf.constant([[[oc_Add, 1, 2], [oc_Sub, 4, 1], [oc_Ret, 5, 0]]])
    oc_data = tf.reshape(tf.slice(program, [0, 0, 0], [-1, -1, 1]), [batch_size, -1])
    print("oc_data: {}".format(oc_data.get_shape())) # [batch_size x max_len]
    
    firstOp_data = tf.reshape(tf.slice(program, [0, 0, 1], [-1, -1, 1]), [batch_size, -1])
    print("firstOp_data: {}".format(firstOp_data.get_shape())) # [batch_size x max_len]
    
    sndOp_data = tf.reshape(tf.slice(program, [0, 0, 2], [-1, -1, 1]), [batch_size, -1])
else:
    # training batch size
    batch_size = 256

    # maximal program len
    max_Time = 32

    # maximal number of parameters
    num_Params = 5

    # input feed
    oc_data = tf.placeholder(tf.int32, [batch_size, max_Time])
    firstOp_data = tf.placeholder(tf.int32, [batch_size, max_Time])
    sndOp_data = tf.placeholder(tf.int32, [batch_size, max_Time])

# valid operand index range for this program
lowestOperand=-2
highestOperand=1

# most basic version -> operate over a chain of op codes (just for testing)
with tf.Session() as sess:
    ### OK
    sess.run(tf.global_variables_initializer())
    if DummyRun:
        print(oc_data.eval())
        print(firstOp_data.eval())
        print(sndOp_data.eval())

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
    
    state = initial_state
    with tf.variable_scope("DAG"): # Recursive Dag Network
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
      output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size], name="output")

    # merged = tf.merge_all_summaries()
    writer = tf.summary.FileWriter("build/tf_logs", sess.graph_def)

    tf.train.write_graph(sess.graph_def, 'build/', 'apo_graph.pb', as_text=False)
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
