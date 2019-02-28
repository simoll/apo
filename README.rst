=================================
APO - Automatic Program Optimizer
=================================

APO is a toy project to explore reinforcement learning for program optimization - pattern-rewriting on compute DAGs in the search for the shortest program.
The project is inspired by AlphaZero (https://arxiv.org/abs/1712.01815).


The toy language
----------------



This toy program computes the expression ``7 * ((b + a) - b)``:
::
    0: add %b %a
    1: sub %0 %b
    2: 7
    3: mul %2 %1
    4: ret %3

A toy program consists of a sequence of operations on unsigned, 32bit integers.
Every statement has the form
::
    <line_no>: <op>( <id)*


*\<id\> - Identifiers*

Identifiers can refer to line numbers or arguments, for example:
``%2`` - is the result of the statement in line ``2`` (general case ``%n`` with ``n`` being an integer ``>= 0``).
``%c`` - is the value of the third parameter (``%a`` to ``%z``).


*\<op\> - Binary Operators*
::
    3: add %0 %1

adds the values of variables ``%0`` and ``%1`` and store the result in ``%3``.
available operators are ``add / sub / mul / and / or / xor``.


*\<op\> - Constant-yielding*
::
    2: 42

Stores ``42`` in variable ``%2``.


*\<op\> - pipe (fake use)*
::
    3: # %2

Assign to ``%3`` the contents of ``%2``.


*\<op\> - return value*
::
    6: ret %5

``%5`` is the return value of the program.


configuration files
-------------------

devices.conf
""""""""""""

*Purpose:*

Tensorflow tower configuration file

*Syntax:*
::
  <tensorflow_device> <name> <task1>[,<task2>]*  <rating>

*Interpretation:*

create a tower on <tensorflow_device> with the op-prefix <name>/. The tower is instantiated for the tasks specified in <taskSet>.
The interpretation of <rating> depends on the task. It is possible to create multiple towers per device.

infer task:
  Inference tower used during reinforcement learning.
  <rating> translates to the number of concurrent search threads using this inference device.
  An infer tower uses a StagingArea (dev/put_stage is required to pull data into the model before dev/infer_X_dist can be used).

loss task:
  Inference tower used for loss computation (logging).
  This device is unbuffered (directly feeded from tf.placeholders).
  <rating> does not have any meaning here.

train task:
  There must be only a single train device at the moment.
  <rating> may in the future be used to automatically distribute gradient computation to all train devices.

server.conf
"""""""""""

SampleServer configuration (mostly the training sample queue).

train.conf
""""""""""

Soft model parameters. Does not affect the MetaGraph so no rebuilding necessary.

model.conf
""""""""""

Hard model parameters. If any entry changes, the Tensorflow MetaGraph needs to be rebuild (which takes forever).




