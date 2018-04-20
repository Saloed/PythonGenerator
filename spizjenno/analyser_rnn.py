from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, nn_impl
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops as ta_ops
from tensorflow.python.ops import variable_scope as vs

from . import dynamic_rnn
from .attention_dynamic_rnn import attention_dynamic_rnn
from .attention_dynamic_rnn import stack_attention_tree_dynamic_rnn, stack_attention_dynamic_rnn
from .dynamic_rnn import bidirectional_dynamic_rnn

# noinspection PyProtectedMember
_WEIGHTS_NAME = dynamic_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = dynamic_rnn._BIAS_NAME


def sequence_input(cell_fw, cell_bw, inputs, inputs_length, hidden_size=None,
                   dtype=None, scope=None):
    batch_size = inputs.get_shape()[0].value
    assert batch_size == inputs_length.get_shape()[0].value, "Batch sizes of inputs and inputs lengths must be equals"
    input_length = array_ops.shape(inputs)[1]
    input_size = inputs.get_shape()[2].value
    if hidden_size is None:
        hidden_size = input_size

    with vs.variable_scope(scope or "SequenceInput", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("InputProjectionVariables"):
            W = vs.get_variable(_WEIGHTS_NAME, [input_size, hidden_size], dtype)
            B = vs.get_variable(_BIAS_NAME, [hidden_size], dtype, bias_initializer)

        inputs = array_ops.reshape(inputs, [batch_size * input_length, input_size])
        hiddens = nn_impl.relu_layer(inputs, W, B)
        hiddens = array_ops.reshape(hiddens, [batch_size, input_length, hidden_size])
        _, states = bidirectional_dynamic_rnn(cell_fw, cell_bw, hiddens, inputs_length, dtype=dtype, time_major=True)
        states = array_ops.concat(states, 2)
    return states


def labels_output(cell, attention, output_size, output_length, initial_state=None,
                  hidden_size=None, loop_function=None, dtype=None, scope=None, time_major=False):
    batch_size = attention.batch_size
    state_size = cell.state_size
    if hidden_size is None:
        hidden_size = cell.output_size

    with vs.variable_scope(scope or "LabelsOutput", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("OutputProjection"):
            W = vs.get_variable(_WEIGHTS_NAME, [state_size, output_size], dtype)
            B = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)

        inputs = array_ops.zeros([output_length, batch_size, hidden_size])
        results = attention_dynamic_rnn(
            cell, inputs, attention, initial_state=initial_state,
            loop_function=loop_function, dtype=dtype)
        outputs, states, attentions, weights = results
        logits = array_ops.reshape(states, [output_length * batch_size, state_size])
        logits = logits @ W + B
        outputs = nn_ops.softmax(logits)
        logits = array_ops.reshape(logits, [output_length, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_length, batch_size, output_size])
        if not time_major:
            logits = array_ops.transpose(logits, [1, 0, 2])
            outputs = array_ops.transpose(outputs, [1, 0, 2])
    return logits, outputs, states, attentions, weights


def sequence_tokens_output(cell, attention, output_size, output_length, initial_states=None, output_height=None,
                           hidden_size=None, loop_function=None, stack_function=None, dtype=None, scope=None):
    batch_size = attention.batch_size
    state_size = cell.state_size
    if hidden_size is None:
        hidden_size = cell.output_size
    assert output_height is not None or initial_states is not None
    if output_height is None:
        output_height = array_ops.shape(initial_states)[0]

    with vs.variable_scope(scope or "SequenceTokensOutput", dtype=dtype) as scope:
        dtype = scope.dtype

        with vs.variable_scope("OutputProjection"):
            W = vs.get_variable(_WEIGHTS_NAME, [state_size, output_size], dtype)
            B = vs.get_variable(_BIAS_NAME, [output_size], dtype, init_ops.constant_initializer(0, dtype))

        inputs = array_ops.zeros([output_length, batch_size, hidden_size])
        results = stack_attention_dynamic_rnn(
            cell, inputs, attention, stack_size=output_height, initial_states=initial_states,
            loop_function=loop_function, stack_function=stack_function, dtype=dtype)
        outputs, states, attentions, weights = results
        logits = array_ops.reshape(states, [output_height * output_length * batch_size, state_size])
        logits = logits @ W + B
        outputs = nn_ops.softmax(logits)
        logits = array_ops.reshape(logits, [output_height, output_length, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_height, output_length, batch_size, output_size])
        logits = array_ops.transpose(logits, [2, 0, 1, 3])
        outputs = array_ops.transpose(outputs, [2, 0, 1, 3])
    return logits, outputs, states, attentions, weights


def tree_tokens_output(cell, attention, output_size, output_length, initial_states=None, output_height=None,
                       hidden_size=None, loop_function=None, stack_function=None, dtype=None, scope=None):
    """

    :param cell:
    :param attention:
    :param output_size:
    :param output_length:
    :param initial_states: 3D-Tensor with shape [output_height x batch_size x ?]
    :param output_height:
    :param hidden_size:
    :param loop_function:
    :param stack_function:
    :param dtype:
    :param scope:
    :return:
    """

    assert isinstance(cell, (list, tuple)) and len(cell) == 2
    state_size = cell[0].state_size
    assert state_size == attention.state_size
    if hidden_size is None:
        hidden_size = cell[0].output_size
    assert output_height is not None or initial_states is not None
    _output_height = array_ops.shape(initial_states)[0]
    if output_height is None:
        output_height = _output_height
    assert initial_states is None or _output_height == output_height
    batch_size = initial_states.get_shape()[1].value
    initial_state_size = initial_states.get_shape()[2].value
    assert batch_size == attention.batch_size

    with vs.variable_scope(scope or "TreeTokensOutput", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        if initial_states is not None:
            with vs.variable_scope("StateProjection"):
                W_state = vs.get_variable(_WEIGHTS_NAME, [initial_state_size, state_size], dtype)
                B_state = vs.get_variable(_BIAS_NAME, [state_size], dtype, bias_initializer)

            shape = [output_height * batch_size, initial_state_size]
            initial_states = array_ops.reshape(initial_states, shape)
            initial_states = nn_impl.relu_layer(initial_states, W_state, B_state)
            shape = [output_height, batch_size, state_size]
            initial_states = array_ops.reshape(initial_states, shape)
        with vs.variable_scope("OutputProjection"):
            W = vs.get_variable(_WEIGHTS_NAME, [state_size, output_size], dtype)
            B = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)

        inputs = array_ops.zeros([output_length, batch_size, hidden_size])
        results = stack_attention_tree_dynamic_rnn(
            cell, inputs, attention, stack_size=output_height, initial_states=initial_states,
            loop_function=loop_function, stack_function=stack_function, dtype=dtype)
        outputs, states, attentions, weights = results
        logits = array_ops.reshape(states, [output_height * output_length * batch_size, state_size])
        logits = logits @ W + B
        outputs = nn_ops.softmax(logits)
        logits = array_ops.reshape(logits, [output_height, output_length, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_height, output_length, batch_size, output_size])
        logits = array_ops.transpose(logits, [2, 0, 1, 3])
        outputs = array_ops.transpose(outputs, [2, 0, 1, 3])
    return logits, outputs, states, attentions, weights


def strings_output(cell, attention, output_size, output_length, initial_states,
                   hidden_size=None, loop_function=None, stack_function=None, dtype=None, scope=None):
    """

    :param cell:
    :param attention:
    :param output_size:
    :param output_length:
    :param initial_states: 4D-Tensor with shape [output_width x output_height x batch_size x ?]
    :param hidden_size:
    :param loop_function:
    :param stack_function:
    :param dtype:
    :param scope:
    :return:
    """

    state_size = cell.state_size
    if hidden_size is None:
        hidden_size = cell.output_size
    output_width = array_ops.shape(initial_states)[0]
    output_height = array_ops.shape(initial_states)[1]
    batch_size = initial_states.get_shape()[2].value
    initial_state_size = initial_states.get_shape()[3].value
    assert batch_size == attention.batch_size

    with vs.variable_scope(scope or "StringsOutput", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("OutputProjection"):
            W = vs.get_variable(_WEIGHTS_NAME, [state_size, output_size], dtype)
            B = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)
        with vs.variable_scope("StateProjection"):
            W_state = vs.get_variable(_WEIGHTS_NAME, [initial_state_size, state_size], dtype)
            B_state = vs.get_variable(_BIAS_NAME, [state_size], dtype, bias_initializer)

        shape = [output_width * output_height * batch_size, initial_state_size]
        initial_states = array_ops.reshape(initial_states, shape)
        initial_states = nn_impl.relu_layer(initial_states, W_state, B_state)
        shape = [output_width, output_height, batch_size, state_size]
        initial_states = array_ops.reshape(initial_states, shape)

        with vs.variable_scope("Arrays"):
            shape = [None, None, batch_size, state_size]
            state_ta = ta_ops.TensorArray(dtype, output_width, None, None, "states", element_shape=shape)
            weights_ta = []
            for i in range(attention.length):
                shape = [None, None, batch_size, None]
                ta = ta_ops.TensorArray(dtype, output_width, None, None, "weights_%d" % i, element_shape=shape)
                weights_ta.append(ta)
            attentions_ta = []
            for i in range(attention.length):
                shape = [None, None, batch_size, attention.input_size]
                ta = ta_ops.TensorArray(dtype, output_width, None, None, "attentions_%d" % i, element_shape=shape)
                attentions_ta.append(ta)

        def time_step(time, state_ta, attentions_ta, weights_ta):
            with vs.variable_scope("TimeStep"):
                inputs = array_ops.zeros([output_length, batch_size, hidden_size])
                _initial_states = array_ops.gather(initial_states, time)
                results = stack_attention_dynamic_rnn(
                    cell, inputs, attention, stack_size=output_height, initial_states=_initial_states,
                    loop_function=loop_function, stack_function=stack_function, dtype=dtype)
                outputs, states, attentions, weights = results
                state_ta = state_ta.write(time, states)
                for i, (attn, weight) in enumerate(zip(attentions, weights)):
                    attentions_ta[i] = attentions_ta[i].write(time, attn)
                    weights_ta[i] = weights_ta[i].write(time, weight)
            return time + 1, state_ta, attentions_ta, weights_ta

        def cond(time, state_ta, attentions_ta, weights_ta):
            return time < output_width

        time = array_ops.constant(0, dtypes.int32, name="time")
        arrays = (state_ta, attentions_ta, weights_ta)
        _, *arrays = control_flow_ops.while_loop(cond, time_step, (time, *arrays))
        state_ta, attentions_ta, weights_ta = arrays
        states = state_ta.stack()
        attentions = [attention_ta.stack() for attention_ta in attentions_ta]
        weights = [weight_ta.stack() for weight_ta in weights_ta]
        logits = array_ops.reshape(states, [output_width * output_height * output_length * batch_size, state_size])
        logits = logits @ W + B
        outputs = nn_ops.softmax(logits)
        logits = array_ops.reshape(logits, [output_width, output_height, output_length, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_width, output_height, output_length, batch_size, output_size])
        logits = array_ops.transpose(logits, [3, 0, 1, 2, 4])
        outputs = array_ops.transpose(outputs, [3, 0, 1, 2, 4])
    return logits, outputs, states, attentions, weights
