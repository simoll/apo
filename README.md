# APO - Automatic Program Optimizer

##

## configuration files
### devices.conf 
Purpose: Tensorflow tower configuration file
Syntax `<tensorflow_device> <name> <task1>[,<task2>]*  <rating>`
Interpretation: create a tower on <tensorflow_device> with the op-prefix <name>/. The tower is instantiated for the tasks specified in <taskSet>.
The interpretation of <rating> depends on the task. It is possible to create multiple towers per device.
-- infer task --
Inference tower used during reinforcement learning.
<rating> translates to the number of concurrent search threads using this inference device.
An infer tower uses a StagingArea (dev/put_stage is required to pull data into the model before dev/infer_X_dist can be used).

-- loss task --
Inference tower used for loss computation (logging).
This device is unbuffered (directly feeded from tf.placeholders).
<rating> does not have any meaning here.

-- train task --
There must be only a single train device at the moment.
<rating> may in the future be used to automatically distribute gradient computation to all train devices.

### server.conf
SampleServer configuration (mostly the training sample queue).

### train.conf
Soft model parameters. Does not affect the MetaGraph so no rebuilding necessary.

### model.conf
Hard model parameters. If any entry changes, the Tensorflow MetaGraph needs to be rebuild (which tasked forever).

