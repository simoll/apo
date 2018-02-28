import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

def data_type():
    return tf.float32




# enable debug output
Debug = False

# set to true for pseudo inputs
Training = True

def parseConfig(fileName):
  res = dict()
  for line in open(fileName, 'r'):
    parts = line.split(" ")
    res[parts[0]] = parts[1]
  return res

### model configuration ###
conf = parseConfig("model.conf")

# maximal program len
prog_length = int(conf["prog_length"])

num_OpCodes = int(conf["max_OpCodes"])

# maximal number of parameters
num_Params = int(conf["num_Params"])

# number of re-write rules
max_Rules = int(conf["max_Rules"]) #, 17

# number of scalar cells in the LSTM
state_size = int(conf["state_size"]) #64

# op code embedding size
embed_size = int(conf["embed_size"]) #32

# stacked cells
num_hidden_layers = int(conf["num_hidden_layers"]) #6

# decoder layers (state wraparound)
num_decoder_layers = int(conf["num_decoder_layers"]) #2

# cell_type
cell_type= conf["cell_type"].strip()
print("Model (construct). num_OpCodes={}, prog_length={}, num_Params={}, max_Rules={}, embed_size={}, state_size={}, num_hidden_layers={}, num_decoder_layers={}, cell_type={}".format(num_OpCodes, prog_length, num_Params, max_Rules, embed_size, state_size, num_hidden_layers, num_decoder_layers, cell_type))
# input feed

# training control parameter (has auto increment)
global_step = tf.get_variable("global_step", initializer = 0, dtype=tf.int32, trainable=False)





with tf.Session() as sess:
    ### OK
    #     sess.run(tf.global_variables_initializer())
    #     print(oc_data.eval())
    #     print(firstOp_data.eval())
    #     print(sndOp_data.eval())


    # one-hot operand encoding
    # firstOp_inputs = tf.one_hot(firstOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())
    # sndOp_inputs = tf.one_hot(smdOp_data, num_Indices,  on_value=0.0, off_value=1, dtype=data_type())

    # parameter input
    param_data = tf.get_variable("param_embed", [num_Params, state_size])

    # devName="/gpu:0"
    def buildTower():
      # shared embedding
      with tf.device("/cpu:0"):
        oc_init = tf.truncated_normal([num_OpCodes, embed_size], dtype=data_type())
        oc_embedding = tf.get_variable("oc_embed", initializer = oc_init, dtype=data_type())

      # number of instructions in the program
      length_data = tf.placeholder(tf.int32, [None], name="length_data")
      batch_size=tf.shape(length_data)[0]
      
      # opCode per instruction
      oc_data = tf.placeholder(tf.int32, [None, prog_length], name="oc_data")
      
      # first operand index per instruction
      firstOp_data = tf.placeholder(tf.int32, [None, prog_length], name="firstOp_data")
      
      # second operand index per instruction
      sndOp_data = tf.placeholder(tf.int32, [None, prog_length], name="sndOp_data")

      oc_inputs = tf.nn.embedding_lookup(oc_embedding, oc_data) # [batch_size x idx x embed_size]
      print("oc_inputs : {}".format(oc_inputs.get_shape())) # [ batch_size x max_len x embed_size ]

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
      if cell_type == "gru":
          make_cell = lambda: tf.nn.rnn_cell.GRUCell(state_size)
      elif cell_type == "lstm_block":
          tupleState = True
          make_cell = lambda: tf.contrib.rnn.LSTMBlockCell(state_size)
      elif cell_type == "lstm":
          tupleState = True
          make_cell = lambda: tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
      else:
          print(cell_type)
          print("Failed to build model! Choose an RNN cell type.")
          raise SystemExit


      initial_outputs = outputs
      
      ### network setup ###
      UseRDN=True 
      if UseRDN:   # Recursive Dag Network
          # TODO document
          with tf.variable_scope("DAG"): 
            out_states=[]
            last_states=[]

            next_initial = None
            for l in range(num_hidden_layers + num_decoder_layers):
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
                # [prog_length x batch_size]
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

                # hidden layers have zero initial states
                if l < num_hidden_layers:
                  # hidden - initial
                  next_initial = cell.zero_state(dtype=data_type(), batch_size=batch_size)
                else:
                  # decoder - initial state
                  state_idx = (l - num_decoder_layers)
                  next_initial = last_states[state_idx]

                outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=next_initial, sequence_length=length_data)
                # next_initial = state # LSTM wrap around for decoder layers

                if tupleState:
                  # e.g. LSTM
                  out_states.append(state[0])
                  out_states.append(state[1])
                  last_states.append(state)
                else:
                  # e.g. GRU
                  out_states.append(state)
                  last_states.append(state)

              # last_output_size = tf.dim_size(outputs[0], 0)

          # last layer output state
          net_out = tf.concat(out_states, axis=1)

      else:
          # multi layer cell
          if num_hidden_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_hidden_layers)], state_is_tuple=True)
          else:
            cell = make_cell()

          initial_state = cell.zero_state(dtype=data_type(), batch_size=batch_size)

          # use a plain LSTM
          inputs = tf.unstack(oc_inputs, num=prog_length, axis=1)
          outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=initial_state, sequence_length=length_data)# swap_memory=True)
          last_output = outputs[-1]
          if num_hidden_layers > 1:
            net_out = tf.reshape(state[-1].c, [batch_size, -1])
          else:
            net_out = state.c

      ### per-node objective ###
      # target_in = tf.placeholder(data_type(), [None, prog_length], name="target_in")        # dist over pcs
      # rule_in = tf.placeholder(data_type(), [None, prog_length, max_Rules], name="dist_in") # dist over pcs x rule

      def make_relu(in_size):
        # target probability [0, 1] per node
        # variables (all [state_size])
        with tf.variable_scope("relu_state"):
          m_init = tf.truncated_normal([in_size, in_size], dtype=data_type())
          v_bias_init = tf.truncated_normal([in_size], dtype=data_type())
          v_project_init = tf.truncated_normal([in_size], dtype=data_type())
          m_trans = tf.get_variable("m_trans", initializer=m_init, trainable=True)
          v_bias = tf.get_variable("v_bias", initializer=v_bias_init, trainable=True)
          v_project = tf.get_variable("v_project", initializer=v_project_init, trainable=True)

        # target probability unit
        def target_unit(batch):
          # dense layer (todo accumulate state)
          # with tf.variable_scope("dense", reuse=len(accu) > 0):
          #   t = tf.layers.dense(inputs=batch, units=1)[0]

          with tf.variable_scope("relu"):
            elem_trans = tf.nn.relu_layer(batch, m_trans, v_bias)
            # elem_trans = tf.nn.relu(tf.matmul(batch, m_trans) + v_bias)
            t = tf.reduce_sum(v_project * elem_trans, axis=1)
          return t

        return target_unit

      def make_linear(in_size):
        # target probability [0, 1] per node
        # variables (all [state_size])
        with tf.variable_scope("linear_state"):
          m_init = tf.truncated_normal([in_size, in_size], dtype=data_type())
          v_bias_init = tf.truncated_normal([in_size], dtype=data_type())
          v_project_init = tf.truncated_normal([in_size], dtype=data_type())
          m_trans = tf.get_variable("m_trans", initializer=m_init, trainable=True)
          v_bias = tf.get_variable("v_bias", initializer=v_bias_init, trainable=True)
          v_project = tf.get_variable("v_project", initializer=v_project_init, trainable=True)

        # target probability unit
        def target_unit(batch):
          # dense layer (todo accumulate state)
          # with tf.variable_scope("dense", reuse=len(accu) > 0):
          #   t = tf.layers.dense(inputs=batch, units=1)[0]

          with tf.variable_scope("linear"):
            elem_trans = tf.matmul(batch, m_trans) + v_bias
            t = tf.reduce_sum(v_project * elem_trans, axis=1)
          return t

        return target_unit


      ### target logits ###
      pool = tf.reduce_sum(outputs, axis=0) #[batch_size]

      with tf.variable_scope("target"):
        cell = make_relu(state_size * 2)
        accu=[]
        for batch in outputs:
          J = tf.concat([batch, pool], axis=1) # TODO factor out pool transformation (redundant)
          with tf.variable_scope("instance", reuse=len(accu) > 0):
            pc_target_logit = tf.layers.dense(inputs=net_out, activation=tf.identity, units=1)[:, 0]
          accu.append(pc_target_logit)
          # accu.append(cell(J))
      # [prog_length x batch_size] -> [batch_size x prog_length]
      target_logits = tf.transpose(accu, [1, 0])

      ### stop logit ###
      # with tf.variable_scope("stop"):
      #   cell = make_linear(state_size)
      #   stop_logit = cell(pool) # based on last layer RNN output state

      # pred_stop_dist = tf.sigmoid(stop_logit, name="pred_stop_dist") # only positive values?????????????
      with tf.variable_scope("stop"):
         stop_logits = tf.layers.dense(inputs=net_out, activation=tf.identity, units=1)[:, 0]

      pred_stop_dist = tf.sigmoid(stop_logits, name="pred_stop_dist")

      ### rule logits ###
      with tf.variable_scope("rules"):
        accu=[]
        for batch in outputs:
          J = tf.concat([batch, pool], axis=1)
          with tf.variable_scope("", reuse=len(accu) > 0):
            rule_bit = tf.layers.dense(inputs=J, activation=tf.identity, units=max_Rules, name="layer")
          accu.append(rule_bit)
      action_logits = tf.transpose(accu, [1, 0, 2]) # [batch_size x prog_length x max_Rules]

      ## predictions ##
      # distributions
      tf.nn.sigmoid(action_logits,name="pred_action_dist")
      tf.nn.sigmoid(target_logits,name="pred_target_dist")

      ### reference input & training ###
      # reference input #
      stop_in = tf.placeholder(data_type(), [None], name="stop_in") # stop indicator
      action_in = tf.placeholder(data_type(), [None, prog_length, max_Rules], name="action_in") # action distribution (over rules per instruction)
      target_in = tf.placeholder(data_type(), [None, prog_length], name="target_in") # target distribution (over instructions (per program))

      # training #
      # stop_loss = tf.losses.absolute_difference(stop_in, pred_stop_dist) # []
      stop_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_in, logits=stop_logits) # []

      num_action_elems = prog_length * max_Rules
      per_action_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(action_in, [-1, num_action_elems]), logits=tf.reshape(action_logits, [-1, num_action_elems])) #, dim=-1) # [batch_size]
      action_loss = tf.reduce_mean(per_action_loss, axis=1)

      target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_in, logits=target_logits), axis=1) # [batch_size]

      # cummulative action loss
      move_losses = action_loss + target_loss 

      # conditional loss (only penalize rule/target if !stop_in)
      loss = tf.reduce_mean((1.0 - stop_in) * move_losses) + stop_loss
      tf.identity(loss, "loss")

      mean_stop_loss = tf.reduce_mean(stop_loss, name="mean_stop_loss")
      mean_action_loss = tf.reduce_mean((1.0 - stop_in) * action_loss, name="mean_action_loss")
      mean_target_loss = tf.reduce_mean((1.0 - stop_in) * target_loss, name="mean_target_loss")

      # return action handles
      return (loss, mean_stop_loss, mean_action_loss, mean_target_loss)
    # END buildTower

    hasTrainDevice = False
    hasInferDevice = False
    laterDevice = None # same as False from Tensorflow 1.1
    trainTower = None # tower used for training

    # create one tower for each device
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      for line in open("devices.conf", 'r'):
        if len(line) == 0 or line[0] == "#":
          continue

        # actual device entry 
        parts = line.split(" ")
        devName = parts[0]     # tensorflow device name, eg "/gpu:0"
        towerName = parts[1]   # tower name to be used in apo, eg "g0"
        taskSet = parts[2]     # task set this device shall be associated with, eg "infer,train"
        rating = int(parts[3]) # device performance rating, eg CPU has 1 , GPU has 10

        # build tower
        with tf.device(devName):
          with tf.variable_scope("net", reuse=laterDevice, auxiliary_name_scope=False):
            with tf.name_scope(towerName):
              devTower = buildTower()

        laterDevice=True
        hasInferDevice = hasInferDevice or ("infer" in taskSet)

        if not "train" in taskSet:
          continue

        # configure training device
        if hasTrainDevice:
          print("Already defined a training device!!!\n")
          raise SystemExit

        hasTrainDevice = True
        devLoss = devTower[0]
        trainTower = towerName


    # attach optimizer to "train" device (defining the unique "train_dist_op")
    if False:
    # learning rate configuration
      starter_learning_rate = 0.001
      # end_learning_rate = 0.0001
      decay_steps = 1000
      decay_rate = 0.99

      learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                global_step,
                                                decay_steps,
                                                decay_rate,
                                                name="learning_rate")
    
    else:
      # learning rate parameter
      learning_rate = tf.get_variable("learning_rate", initializer=0.001, dtype=tf.float32, trainable=False)
      
      # set_learning_rate op (to set learning_rate from APO)
      new_learning_rate = tf.placeholder(tf.float32, [], "new_learning_rate")
      tf.assign(learning_rate, new_learning_rate, name="set_learning_rate")

    with tf.device(devName):
      # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # seems to perform better on the "count-oc_Add-task"

      train_dist_op = optimizer.minimize(
          loss=devLoss,
          global_step=global_step,
          name="train_dist_op")

    # tf.summary.scalar('loss', cpuLoss[0])

    ### prob of getting the cout right (pCorrect_op)  ###

    # def equals_fn(x,y):
    #   return 1 if x == y else 0

    # matched_rule = tf.cast(tf.equal(pred_rule, rule_in), tf.float32) #tf.map_fn(equals_fn, zip(predicted, rule_in), dtype=tf.int32, back_prop=false)
    # matched_target = tf.cast(tf.equal(pred_target, target_in), tf.float32) #tf.map_fn(equals_fn, zip(predicted, rule_in), dtype=tf.int32, back_prop=false)
    # pCorrect = tf.reduce_mean([matched_rule, matched_target], name="pCorrect_op")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("build/tf_logs", sess.graph)


    tf.global_variables_initializer().run()
    init = tf.variables_initializer(tf.global_variables(), name='init_op')
    fileName = "apo_graph.pb"
    modelPrefix ="build/rdn"

    # save metagraph
    tf.train.Saver(tf.trainable_variables()).save(sess, modelPrefix) 

    # tf.train.write_graph(sess.graph, 'build/', fileName, as_text=False)
    print("Model written to {}.".format(modelPrefix))
    writer.close()
    raise SystemExit
