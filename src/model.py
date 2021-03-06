import tensorflow.contrib.staging
from tensorflow.python.util import nest

import tensorflow as tf
import numpy as np
from bnlstm import *

# dtype used for the model and for reference inputs (distribs)
def data_type():
    return tf.float32

# dtype used for the program encoding (prog_length, firstOp_data, ...)
def int_type():
    return tf.int32



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

train_batch_size = int(conf["train_batch_size"])

modelOperands=int(conf["model_operands"]) != 0
modelUsers=int(conf["model_users"]) != 0

# cell_type
cell_type= conf["cell_type"].strip()
print("Model (construct). num_OpCodes={}, prog_length={}, num_Params={}, max_Rules={}, embed_size={}, state_size={}, num_hidden_layers={}, num_decoder_layers={}, cell_type={}".format(num_OpCodes, prog_length, num_Params, max_Rules, embed_size, state_size, num_hidden_layers, num_decoder_layers, cell_type))
# input feed






with tf.Session() as sess:
  # shared embeddings
  with tf.device("/cpu:0"):
    param_init = tf.truncated_normal([num_Params, state_size], dtype=data_type())
    param_embed = tf.get_variable("param_embed", initializer = param_init, dtype=data_type())
    oc_init = tf.truncated_normal([num_OpCodes, embed_size], dtype=data_type())
    oc_embedding = tf.get_variable("oc_embed", initializer = oc_init, dtype=data_type())

  # handle to encoded inputs
  class IRInputs:
    def __init__(self):
      # number of instructions in the program
      self.length_data = None # tf.placeholder(tf.int32, [None], name="length_data")
      
      # opCode per instruction
      self.oc_data = None #tf.placeholder(tf.int32, [None, prog_length], name="oc_data")
      
      # first operand index per instruction
      self.firstOp_data = None #tf.placeholder(tf.int32, [None, prog_length], name="firstOp_data")
      
      # second operand index per instruction
      self.sndOp_data = None #tf.placeholder(tf.int32, [None, prog_length], name="sndOp_data")

  class ReferenceInputs:
    def __init__(self):
      self.stop_in = None
      self.target_in = None
      self.action_in = None
        
  # the output operations of the tower
  class TowerOutputs:
    def __init__(self, action_logits, pred_action_dist, stop_logits, pred_stop_dist, target_logits, pred_target_dist):
      self.action_logits=action_logits
      self.target_logits=target_logits
      self.stop_logits=stop_logits
      self.pred_action_dist=pred_action_dist
      self.pred_stop_dist=pred_stop_dist
      self.pred_target_dist=pred_target_dist

  # loss collection (from a tower)
  class Losses:
    def __init__(self, loss, mean_action_loss, mean_stop_loss, mean_target_loss):
      self.loss = loss
      self.mean_action_loss = mean_action_loss
      self.mean_stop_loss = mean_stop_loss
      self.mean_target_loss = mean_target_loss

  # feeding queue
  class QueuedTrainingInputs:
    def __init__(self, IR, Ref, capacity=2, batch_size=None):
      # shapes (Queue::dequeue is shape oblivious)
      self.length_shape = tf.TensorShape([batch_size])
      self.oc_shape = tf.TensorShape([batch_size, prog_length])
      self.ops_shape = tf.TensorShape([batch_size, prog_length])
      self.stop_shape = tf.TensorShape([batch_size])
      self.target_shape = tf.TensorShape([batch_size, prog_length])
      self.action_shape = tf.TensorShape([batch_size, prog_length, max_Rules])

      with tf.device("/cpu:0"):
        self.length_queue = tf.FIFOQueue(capacity, int_type(), [self.length_shape] if batch_size else None)
        self.oc_queue = tf.FIFOQueue(capacity, int_type(), [self.oc_shape] if batch_size else None)
        self.firstOp_queue = tf.FIFOQueue(capacity, int_type(), [self.ops_shape] if batch_size else None)
        self.sndOp_queue = tf.FIFOQueue(capacity, int_type(), [self.ops_shape] if batch_size else None)
        self.stop_queue = tf.FIFOQueue(capacity, data_type(), [self.stop_shape] if batch_size else None)
        self.target_queue = tf.FIFOQueue(capacity, data_type(), [self.target_shape] if batch_size else None) 
        self.action_queue = tf.FIFOQueue(capacity, data_type(), [self.action_shape] if batch_size else None) 

        self.length_queue.enqueue(IR.length_data, "q_length_data")
        self.oc_queue.enqueue(IR.oc_data, "q_oc_data")
        self.firstOp_queue.enqueue(IR.firstOp_data, "q_firstOp_data")
        self.sndOp_queue.enqueue(IR.sndOp_data, "q_sndOp_data")

        # from ReferenceInputs
        self.stop_queue.enqueue(Ref.stop_in, "q_stop_in")

        self.target_queue.enqueue(Ref.target_in, "q_target_in")

        self.action_queue.enqueue(Ref.action_in, "q_action_in")

      # staging area (device scope)
      self.stage = tf.contrib.staging.StagingArea( \
          [int_type(), int_type(), int_type(), int_type(), data_type(), data_type(), data_type()], \
          [self.length_shape, self.oc_shape, self.ops_shape, self.ops_shape, self.stop_shape, self.target_shape, self.action_shape] if batch_size else None, \
          capacity = capacity)

      # link Queue::dequeue -> StagingArea::put
      IR = IRInputs()
      Ref = ReferenceInputs()

      # dequeue *MUST* happen on CPU
      with tf.device("/cpu:0"):
        #IR (temporary)
        IR.length_data = self.length_queue.dequeue()
        IR.length_data.set_shape(self.length_shape)

        IR.oc_data = self.oc_queue.dequeue()
        IR.oc_data.set_shape(self.oc_shape)

        IR.firstOp_data = self.firstOp_queue.dequeue()
        IR.firstOp_data.set_shape(self.ops_shape)

        IR.sndOp_data = self.sndOp_queue.dequeue()
        IR.sndOp_data.set_shape(self.ops_shape)

        # Ref (temporary)_
        Ref.stop_in = self.stop_queue.dequeue()
        Ref.stop_in.set_shape(self.stop_shape)

        Ref.action_in = self.action_queue.dequeue()
        Ref.action_in.set_shape(self.action_shape)

        Ref.target_in = self.target_queue.dequeue()
        Ref.target_in.set_shape(self.target_shape)

        # transfer to stage
        self.stage.put((IR.length_data, IR.oc_data, IR.firstOp_data, IR.sndOp_data, Ref.stop_in, Ref.target_in, Ref.action_in), name="forward_stage")

    # dequeue inputs to bs used for training
    def dequeue(self):
      IR = IRInputs()
      Ref = ReferenceInputs()

      (IR.length_data, IR.oc_data, IR.firstOp_data, IR.sndOp_data, Ref.stop_in, Ref.target_in, Ref.action_in) = self.stage.get()

      return IR, Ref

  def buildReferencePlaceholders(batch_size=None):
    Ref = ReferenceInputs()
    Ref.stop_in = tf.placeholder(data_type(), [batch_size], name="stop_in") # stop indicator
    Ref.action_in = tf.placeholder(data_type(), [batch_size, prog_length, max_Rules], name="action_in") # action distribution (over rules per instruction)
    Ref.target_in = tf.placeholder(data_type(), [batch_size, prog_length], name="target_in") # target distribution (over instructions (per program))
    return Ref

  # build an unit capacity inference stage for bulk buffer transfers to the device
  # @return the staged tensors (stage::get)
  def buildInferStage(IR, batch_size=None):
    # shapes (Queue::dequeue is shape oblivious)
    length_shape = tf.TensorShape([batch_size])
    oc_shape = tf.TensorShape([batch_size, prog_length])
    ops_shape = tf.TensorShape([batch_size, prog_length])

    # staging area (device scope)
    stage = tf.contrib.staging.StagingArea( \
        [int_type(), int_type(), int_type(), int_type()], \
        [length_shape, oc_shape, ops_shape, ops_shape] if batch_size else None)

    # transfer to stage
    put_op = stage.put((IR.length_data, IR.oc_data, IR.firstOp_data, IR.sndOp_data), name="put_stage")
    StagedIR = IRInputs()
    StagedIR.length_data, StagedIR.oc_data, StagedIR.firstOp_data, StagedIR.sndOp_data = stage.get(name="get_stage")
    # provide elided shape info
    StagedIR.length_data.set_shape(length_shape)
    StagedIR.oc_data.set_shape(oc_shape)
    StagedIR.firstOp_data.set_shape(ops_shape)
    StagedIR.sndOp_data.set_shape(ops_shape)

    return StagedIR

  # generate and assign placeholder based inputs
  def buildIRPlaceholders(batch_size=None):
    IR = IRInputs()

    # number of instructions in the program
    IR.length_data = tf.placeholder(int_type(), [batch_size], name="length_data")
    
    # opCode per instruction
    IR.oc_data = tf.placeholder(int_type(), [batch_size, prog_length], name="oc_data")
    
    # first operand index per instruction
    IR.firstOp_data = tf.placeholder(int_type(), [batch_size, prog_length], name="firstOp_data")
    
    # second operand index per instruction
    IR.sndOp_data = tf.placeholder(int_type(), [batch_size, prog_length], name="sndOp_data")
    return IR

  # returns @TowerOutputs with the tower's output operations
  def buildTower(IR, is_training=False, batch_size=None):
    if batch_size is None:
      # configure for a dynamic batch_size
      batch_size=tf.shape(IR.length_data)[0]
     
    oc_inputs = tf.nn.embedding_lookup(oc_embedding, IR.oc_data) # [batch_size x idx x embed_size]
    print("oc_inputs : {}".format(oc_inputs.get_shape())) # [ batch_size x max_len x embed_size ]


    # param_batch = tf.split(tf.tile(param_embed, [1, batch_size]), 1, batch_size)
    # print(param_embed.get_shape()) # [numParams x state_size]
    # print(param_batch.get_shape()) # [batch_size x numParams x state_size]

    # attach neutral and param matrices
    # outputs = [zero_state] + param_embed # [batch_size x time x state_size]

    ### recurrent cell setup ###
    tupleState=False
    isFusedCell=False
    if cell_type == "gru":
        make_cell = lambda _: tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == "lstm_block":
        tupleState = True
        make_cell = lambda _: tf.contrib.rnn.LSTMBlockCell(state_size)
    elif cell_type == "lstm_fused":
        tupleState = True
        isFusedCell = True
        make_cell = lambda _: tf.contrib.rnn.LSTMBlockFusedCell(state_size)
    elif cell_type == "lstm":
        tupleState = True
        make_cell = lambda _: tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
    elif cell_type == "bnlstm":
        tupleState = True
        make_cell = lambda is_training: BNLSTMCell(state_size, is_training)
    else:
        print(cell_type)
        print("Failed to build model! Choose an RNN cell type.")
        raise SystemExit

    zero_state = tf.zeros([batch_size, state_size], dtype=data_type())

    ### assemble the common (input) prefix (params, zeros)
    # layout the param embeddings in ascending order
    param_batch = tf.reshape(tf.tile(param_embed, [batch_size, 1]), [batch_size, num_Params, -1])

    # pre-pend zeros for neutral operand (n/a operand position)
    if Debug:
        print("param_batch: {}".format(param_batch.get_shape()))

    seq_prefix = [zero_state]
    for i in range(num_Params):
        seq_prefix.append(param_batch[:, i, :])
    if Debug:
        print(seq_prefix)

    seq_prefix = tf.stack(seq_prefix)
    
    ### network setup ###
    with tf.variable_scope("DAG"): 
      out_states=[]
      last_states=[]

      next_initial = None
      for l in range(num_hidden_layers + num_decoder_layers):

        print("- buildling layer {}".format(l))
        ### assemble layer inputs (based on operands, user and output of last layer)
        with tf.variable_scope("inputs_{}".format(l)):
          print("\t- assemble inputs")
          if l == 0:
            # apply LSTM to opCodes
            inputs = tf.unstack(oc_inputs, num=prog_length, axis=1)
          else:
            transOut = tf.transpose(outputs, perm=[1, 0, 2]) # [batch_size x prog_length x state]
            # bspOut = tf.transpose(outputs, perm=[1, 2, 0]) # [B x S x P]

            # pre-pend common input prefix
            sequence = tf.concat([seq_prefix, outputs], 0) # [P x B x S]

            # next layer inputs to assemble
            inputs=[]
            batch_range = tf.expand_dims(tf.range(0, batch_size), axis=1) # [B x 1] -> batch_range[i][0] == i
            # [prog_length x batch_size]

            # TODO remove loop entirely (rely on tensor operations intead)
            for time_step in range(prog_length):
              # fetch current inputs
              # gather first operand outputs
              if modelOperands:
                def reduceOperands(time_step, opIndices):
                  indices = tf.expand_dims(opIndices[:, time_step], axis=1) # [B x 1]
                  idx = tf.concat([indices, batch_range], axis=1) # [B x 2] , indexes into P and B of sequence
                  return tf.gather_nd(sequence, idx) # [B x S]

                with tf.variable_scope("firstOp"):
                  flat_first = reduceOperands(time_step, IR.firstOp_data)

                # gather second operand outputs
                with tf.variable_scope("sndOp"):
                  flat_snd = reduceOperands(time_step, IR.sndOp_data)

              else:
                flat_first = None
                flat_snd = None

              if modelUsers:
                def reduceUsers(time_step, userIndices):
                  # filter out all users of the current pc (time_step + num_Params)
                  userMask = tf.equal(userIndices, 1 + num_Params + time_step) # [batch_size x prog_length]
                  if True:
                    # faster variant (2.4s)
                    expUserMask = tf.tile(tf.expand_dims(userMask, axis=2), [1, 1, state_size])

                    # mask user states
                    userStates = tf.where(expUserMask, transOut, tf.zeros_like(transOut, dtype=data_type()), name="userStates")
                    # userStates : [batch_size x prog_length x state]

                    userInput = tf.reduce_sum(userStates, axis=1) # [batch_size x state]
                  else:
                    # slower variant (3.3s)
                    userMat = tf.expand_dims(tf.where(userMask, tf.ones_like(userMask, dtype=data_type()), tf.zeros_like(userMask, dtype=data_type())), axis=2) # [B x P x 1]
                    userSum = tf.matmul(bspOut, userMat) # [B x S x P] * [B x P x 1] -> [B x S x 1]
                    userInput = userSum[:, :, 0] # [B x S]

                  return userInput

                with tf.variable_scope("users"):
                  first_user_state = reduceUsers(time_step, IR.firstOp_data)
                  snd_user_state = reduceUsers(time_step, IR.sndOp_data)
                  user_input = first_user_state + snd_user_state # [batch_size x state]

              else:
                user_input = None

              # merge into input tensor
              elems = [outputs[time_step]]
              if modelOperands:
                elems.append(flat_first)
                elems.append(flat_snd)
              if modelUsers:
                elems.append(user_input)

              time_input = tf.concat(elems, axis=1, name="cell_input")
              inputs.append(time_input)

        ### instantiate an RNN layer on the inputs
        print("\t- roll out RNN")
        with tf.variable_scope("layer_{}".format(l)): # Recursive Dag Network
          cell = make_cell(is_training)

          zero_cell_state = cell.zero_state(batch_size=batch_size, dtype = data_type()) if not isFusedCell else (zero_state, zero_state) 

          # hidden layers have zero initial states
          if l < num_hidden_layers:
            # hidden - initial
            next_initial = zero_cell_state
          else:
            # decoder - initial state
            state_idx = (l - num_decoder_layers)
            next_initial = last_states[state_idx]

          if isFusedCell:
             # [batch_size x time x input_size]
             outputs, state = cell(tf.stack(inputs), initial_state=next_initial, sequence_length=IR.length_data)
          else:
             outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=next_initial, sequence_length=IR.length_data)

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
    print("- target logits")
    pool = tf.reduce_sum(outputs, axis=0) #[batch_size]

    with tf.variable_scope("target"):
      cell = make_relu(state_size * 2)
      accu=[]
      for i in range(prog_length):
        batch = outputs[i]
      #for batch in outputs:
        J = tf.concat([batch, pool], axis=1) # TODO factor out pool transformation (redundant)
        with tf.variable_scope("instance", reuse=len(accu) > 0):
          pc_target_logit = tf.layers.dense(inputs=net_out, activation=tf.identity, units=1)[:, 0]
        accu.append(pc_target_logit)
        # accu.append(cell(J))
    # [prog_length x batch_size] -> [batch_size x prog_length]
    target_logits = tf.transpose(accu, [1, 0])

    ### stop logit ###
    print("- stop logits")
    with tf.variable_scope("stop"):
       stop_logits = tf.layers.dense(inputs=net_out, activation=tf.identity, units=1)[:, 0]

    ### rule logits ###
    print("- action logits")
    with tf.variable_scope("actions"):
      accu=[]
      for i in range(prog_length):
        batch = outputs[i]
      # for batch in outputs:
        J = tf.concat([batch, pool], axis=1)
        with tf.variable_scope("", reuse=len(accu) > 0):
          rule_bit = tf.layers.dense(inputs=J, activation=tf.identity, units=max_Rules, name="layer")
        accu.append(rule_bit)
    action_logits = tf.transpose(accu, [1, 0, 2]) # [batch_size x prog_length x max_Rules]

    ## predictions ##
    print("- outputs (action, target, stop)")
    pred_stop_dist = tf.sigmoid(stop_logits, name="pred_stop_dist")
    pred_action_dist = tf.nn.sigmoid(action_logits,name="pred_action_dist")
    pred_target_dist = tf.nn.sigmoid(target_logits,name="pred_target_dist")

    # return action handles
    return TowerOutputs(action_logits, pred_action_dist, stop_logits, pred_stop_dist, target_logits, pred_target_dist)
  # END buildTower


  def buildLosses(towerOut, Ref):
    ### reference input & training ###
    # reference input #
    # training #
    # stop_loss = tf.losses.absolute_difference(stop_in, pred_stop_dist) # []
    stop_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Ref.stop_in, logits=towerOut.stop_logits) # []

    num_action_elems = prog_length * max_Rules

    # AlphaZero style: difference to reference distribution
    per_action_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(Ref.action_in, [-1, num_action_elems]), logits=tf.reshape(towerOut.action_logits, [-1, num_action_elems])) #, dim=-1) # [batch_size]
    action_loss = tf.reduce_mean(per_action_loss, axis=1)

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Ref.target_in, logits=towerOut.target_logits), axis=1) # [batch_size]

    # cummulative action loss
    move_losses = action_loss + target_loss 

    # conditional loss (only penalize rule/target if !stop_in)
    loss = tf.reduce_mean((1.0 - Ref.stop_in) * move_losses + stop_loss)
    tf.identity(loss, "loss")

    # mean losses (for reporting only)
    mean_stop_loss = tf.reduce_mean(stop_loss, name="mean_stop_loss")
    mean_action_loss = tf.reduce_mean((1.0 - Ref.stop_in) * action_loss, name="mean_action_loss")
    mean_target_loss = tf.reduce_mean((1.0 - Ref.stop_in) * target_loss, name="mean_target_loss")

    return Losses(loss, mean_action_loss, mean_stop_loss, mean_target_loss)


  # configure global variables (global_step, learning_rate)
  # training control parameter (has auto increment)
  global_step = tf.get_variable("global_step", initializer = 0, dtype=tf.int32, trainable=False)

  # learning_rate
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


  # start buildling the towers
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

      print("Building tower {} for device {} with task set {}".format(towerName, devName, taskSet))

      # completely asynchronous training tower (apo host maintains looper thread)
      isTrainTower = "train" in taskSet
      # tower with staged inference input
      isInferenceTower = "infer" in taskSet
      # constitutes an un-buffered loss tower
      isLossTower = "loss" in taskSet

      if isInferenceTower:
        hasInferDevice = True
        # build an inference tower
        with tf.device(devName):
          with tf.variable_scope("net", reuse=laterDevice, auxiliary_name_scope=False):
            with tf.name_scope(towerName):
              IR = buildIRPlaceholders()
              # StagedIR = buildInferStage(IR) # under test - no benefit in using an inference stage
              towerOut = buildTower(IR, False)

      elif isLossTower:
        hasLossDevice = True
        # build an inference tower
        with tf.device(devName):
          with tf.variable_scope("net", reuse=laterDevice, auxiliary_name_scope=False):
            with tf.name_scope(towerName):
              IR = buildIRPlaceholders()
              Ref = buildReferencePlaceholders()
              towerOut = buildTower(IR, False)

              # make loss inference available
              buildLosses(towerOut, Ref)


      elif isTrainTower:
        # build an training tower
        if hasTrainDevice:
          print("multiple training devices not supported yet!!!\n")
          raise SystemExit

        hasTrainDevice = True

        with tf.device(devName):
          with tf.variable_scope("net", reuse=laterDevice, auxiliary_name_scope=False):
            with tf.name_scope(towerName):
              # build placeholders
              IR = buildIRPlaceholders(batch_size=train_batch_size)
              Ref = buildReferencePlaceholders(batch_size=train_batch_size)

              # attach queues 
              Q = QueuedTrainingInputs(IR, Ref, batch_size=train_batch_size) # must be placed on train device
              QIR, QRef = Q.dequeue()

              towerOut = buildTower(QIR, is_training=True, batch_size=train_batch_size)
              towerLoss = buildLosses(towerOut, QRef)

              # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
              optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # seems to perform better on the "count-oc_Add-task"

            # place this in global name scope (there is only a single training device atm)
            train_dist_op = optimizer.minimize(
                loss=towerLoss.loss,
                global_step=global_step,
                name="train_dist_op")

      laterDevice=True # for tf's reuse flag

  # save the MetaGraph
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
