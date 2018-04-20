from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, math_ops, nn_impl
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops as ta_ops
from tensorflow.python.ops import variable_scope as vs

from . import dynamic_rnn

# noinspection PyProtectedMember
_WEIGHTS_NAME = dynamic_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = dynamic_rnn._BIAS_NAME


def _log2(x, name=None):
    x = math_ops.cast(x, dtypes.float32)
    power = math_ops.log(x)
    base = math_ops.log(2.0)
    return math_ops.div(power, base, name)


class Attention:
    def __init__(self, inputs, state_size, num_heads=1, dtype=None, scope=None):
        """Attention Mechanism for attention decoder RNN.

        :param inputs: list of 3D Tensors or 3D Tensor [batch_size x input_length x input_size].
        :param state_size: Size of the state vectors.
        :param num_heads: Number of attention heads that read from attention_states.
        :param dtype: The dtype to use for the RNN initial state (default: tf.float32).
        :param scope: VariableScope for the created sub-graph; default: "Attention".
        """
        self._state_size = state_size
        if not isinstance(inputs, list):
            inputs = [inputs]
        self._inputs = inputs
        self._num_heads = num_heads
        assert self._num_heads > 0, "With less than 1 heads, not use a attention."
        self._num_inputs = len(self._inputs)
        assert self._num_inputs > 0, "With less than 1 inputs, not use a attention."
        self._input_size = None
        self._batch_size = None
        for input in self._inputs:
            input_size = input.get_shape()[2].value
            assert input_size is not None, "Shape[2] of inputs must be known: %s" % input.get_shape()
            assert self._input_size is None or self._input_size == input_size, "Shape[2] of inputs must be equals for any inputs"
            self._input_size = input_size
            batch_size = input.get_shape()[0].value
            assert batch_size is not None, "Shape[0] of inputs must be known: %s" % input.get_shape()
            assert self._batch_size is None or self._batch_size == batch_size, "Shape[0] of inputs must be equals for any inputs"
            self._batch_size = batch_size

        with vs.variable_scope(scope or "Attention", dtype=dtype) as scope:
            self._scope = scope
            self._dtype = scope.dtype
            self._hiddens_features = []
            self._vectors = []
            self._hiddens = []
            self._inputs_lengths = []
            for i, input in enumerate(self._inputs):
                with vs.variable_scope("AttentionHidden_%d" % i):
                    input_length = input.get_shape()[1].value
                    if input_length is None:
                        input_length = array_ops.shape(input)[1]

                    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
                    hidden = array_ops.reshape(input, [self._batch_size, input_length, 1, self._input_size])

                    hidden_features = []
                    vector = []
                    for j in range(self._num_heads):
                        kernel = vs.get_variable("AttentionW_%d" % j, [1, 1, self._input_size, self._input_size])
                        hidden_features.append(nn_ops.conv2d(hidden, kernel, [1, 1, 1, 1], "SAME"))
                        vector.append(vs.get_variable("AttentionV_%d" % j, [self._input_size]))

                self._hiddens_features.append(hidden_features)
                self._vectors.append(vector)
                self._hiddens.append(hidden)
                self._inputs_lengths.append(input_length)

            bias_initializer = init_ops.constant_initializer(0, self._dtype)
            with vs.variable_scope("AttentionStateProjection"):
                self._W = vs.get_variable(_WEIGHTS_NAME, [state_size, self._input_size], self._dtype)
                self._B = vs.get_variable(_BIAS_NAME, [self._input_size], self._dtype, bias_initializer)

    def call(self, state):
        """Put attention masks on hidden using hidden_features and query."""
        with vs.variable_scope(self._scope or "Attention"):
            attentions = []  # Results of attention reads will be stored here.
            weights = []
            for i in range(self._num_inputs):
                input_length = self._inputs_lengths[i]
                hidden = self._hiddens[i]
                for j in range(self._num_heads):
                    vector = self._vectors[i][j]
                    hidden_features = self._hiddens_features[i][j]
                    y = state @ self._W + self._B
                    y = array_ops.reshape(y, [self._batch_size, 1, 1, self._input_size])
                    # Attention mask is a softmax of v' * tanh(...).
                    s = math_ops.reduce_sum(vector * math_ops.tanh(hidden_features + y), [2, 3])
                    weight = math_ops.sigmoid(s)
                    weights.append(weight)
                    # Now calculate the attention-weighted vector.
                    weight = array_ops.reshape(weight, [self._batch_size, input_length, 1, 1])
                    attention = math_ops.reduce_sum(weight * hidden, [1, 2])
                    attention = array_ops.reshape(attention, [self._batch_size, self._input_size])
                    attentions.append(attention)
        return attentions, weights

    def __call__(self, state):
        return self.call(state)

    @property
    def scope(self):
        return self._scope

    @property
    def state_size(self):
        return self._state_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_size(self):
        return self._input_size

    def input_length(self, i):
        return self._inputs_lengths[i]

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def length(self):
        return self.num_heads * self.num_inputs

    @property
    def output_size(self):
        return self.input_size * self.length


class LoopFunction:
    def __init__(self, input_size, output_size=None, state_size=None, use_input=True, dtype=None, scope=None):
        """
        :param input_size: is a inputs size of rnn unit
        :param output_size: is a output size of rnn unit, if this param is None then output not using for projection
        :param state_size: is a state size of rnn unit, if this param is None then state not using for projection
        :param use_input: if this param is True then input using for projection
        :param dtype: The dtype to use for the projection variables (default: tf.float32).
        :param scope: VariableScope for the created sub-graph; default: "LoopFunction".
        :return: loop_function for connection rnn input with rnn output
        """
        self._input_size = input_size
        self._output_size = output_size
        self._state_size = state_size
        self._use_input = use_input
        assert use_input or output_size is not None or state_size is not None
        hidden_size = (input_size if use_input else 0) + (output_size or 0) + (state_size or 0)

        with vs.variable_scope(scope or "LoopFunction", dtype=dtype) as scope:
            self._scope = scope
            dtype = self._scope.dtype

            bias_initializer = init_ops.constant_initializer(0, dtype)
            with vs.variable_scope("OutputProjection"):
                self._W = vs.get_variable(_WEIGHTS_NAME, [hidden_size, input_size], dtype)
                self._B = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)

    def call(self, input, prev_output, prev_state, time):
        assert self._input_size == input.get_shape()[1].value
        assert self._output_size is None or self._output_size == prev_output.get_shape()[1].value
        assert self._state_size is None or self._state_size == prev_state.get_shape()[1].value

        with vs.variable_scope(self._scope or "LoopFunction"):
            if self._output_size is None:
                result = prev_state
            elif self._state_size is None:
                result = prev_output
            elif self._use_input:
                result = array_ops.concat((input, prev_output, prev_state), 1)
            else:
                result = array_ops.concat((prev_output, prev_state), 1)
            result = nn_impl.relu_layer(result, self._W, self._B)
        return result

    def __call__(self, input, prev_output, prev_state, time):
        return self.call(input, prev_output, prev_state, time)


class StackFunction:
    def __init__(self, input_size, output_size=None, state_size=None, use_inputs=False, dtype=None, scope=None):
        """
        :param input_size: is a inputs size of rnn unit
        :param output_size: is a output size of rnn unit, if this param is None then outputs not using for projection
        :param state_size: is a state size of rnn unit, if this param is None then states not using for projection
        :param use_inputs: if this param is False then inputs not using for projection
        :param dtype: The dtype to use for the projection variables (default: tf.float32).
        :param scope: VariableScope for the created sub-graph; default: "StackFunction".
        :return: stack_function for connection rnn input with rnn output
        """
        self._input_size = input_size
        self._output_size = output_size
        self._state_size = state_size
        self._use_inputs = use_inputs
        assert use_inputs or output_size is not None or state_size is not None
        hidden_size = (input_size if use_inputs else 0) + (output_size or 0) + (state_size or 0)

        with vs.variable_scope(scope or "StackFunction", dtype=dtype) as scope:
            self._scope = scope
            dtype = self._scope.dtype

            bias_initializer = init_ops.constant_initializer(0, dtype)
            with vs.variable_scope("OutputProjection"):
                self._W = vs.get_variable(_WEIGHTS_NAME, [hidden_size, input_size], dtype)
                self._B = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)

    def call(self, inputs, outputs, states, time):
        assert self._input_size == inputs.get_shape()[1].value
        assert self._output_size is None or self._output_size == outputs.get_shape()[1].value
        assert self._state_size is None or self._state_size == states.get_shape()[1].value

        with vs.variable_scope(self._scope or "StackFunction"):
            if self._output_size is None:
                result = states
            elif self._state_size is None:
                result = outputs
            elif self._use_inputs:
                result = array_ops.concat((inputs, outputs, states), 1)
            else:
                result = array_ops.concat((outputs, states), 1)
            result = nn_impl.relu_layer(result, self._W, self._B)
        return result

    def __call__(self, inputs, outputs, states, time):
        return self.call(inputs, outputs, states, time)


def attention_tree_dynamic_rnn(cell, inputs, attention, output_size=None,
                               initial_state=None, loop_function=None,
                               initial_state_attention=True, dtype=None, scope=None):
    assert isinstance(cell, (list, tuple)) and len(cell) == 2
    assert inputs.get_shape()[1].value == attention.batch_size
    assert attention.state_size == cell[0].state_size
    assert attention.state_size == cell[1].state_size
    assert cell[0].output_size == cell[1].output_size
    assert initial_state is None or initial_state.get_shape()[1].value == attention.state_size
    left_cell, right_cell = cell
    batch_size = inputs.get_shape()[1].value
    input_size = inputs.get_shape()[2].value
    state_size = attention.state_size
    if output_size is None:
        output_size = left_cell.output_size
    input_length = array_ops.shape(inputs)[0]
    height = math_ops.cast(math_ops.ceil(_log2(input_length)), dtypes.int32)
    output_length = 2 ** height - 1
    time_steps = 2 ** (height - 1) - 1

    with vs.variable_scope(scope or "AttentionTreeDynamicRnn", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("AttentionInputProjection"):
            W_inp = vs.get_variable(_WEIGHTS_NAME, [input_size + attention.output_size, input_size], dtype)
            B_inp = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)
        with vs.variable_scope("AttentionOutputProjection"):
            W_out = vs.get_variable(_WEIGHTS_NAME, [left_cell.output_size + attention.output_size, output_size], dtype)
            B_out = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)

        with vs.variable_scope("Arrays"):
            shape = [batch_size, state_size]
            state_ta = ta_ops.TensorArray(dtype, output_length, None, False, "states", element_shape=shape)
            shape = [batch_size, output_size]
            output_ta = ta_ops.TensorArray(dtype, output_length, None, False, "outputs", element_shape=shape)
            weights_ta = []
            for i in range(attention.length):
                shape = [batch_size, None]
                ta = ta_ops.TensorArray(dtype, output_length, None, None, "weights_%d" % i, element_shape=shape)
                weights_ta.append(ta)
            attentions_ta = []
            for i in range(attention.length):
                shape = [batch_size, attention.input_size]
                ta = ta_ops.TensorArray(dtype, output_length, None, False, "attentions_%d" % i, element_shape=shape)
                attentions_ta.append(ta)

        def sub_time_step(cell, time, size, state_ta, output_ta, attentions_ta, weights_ta, initial_step=False):
            input = array_ops.gather(inputs, size)
            if initial_step:
                state = initial_state
                attentions = initial_attentions
            else:
                state = state_ta.read(time)
                attentions = [attention_ta.read(time) for attention_ta in attentions_ta]
            if loop_function is not None:
                output = output_ta.read(time)
                input = loop_function(input, output, state, time)
            # If loop_function is set, we use it instead of decoder_inputs.
            x = array_ops.concat([input] + attentions, 1) @ W_inp + B_inp
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            attentions, weights = attention(state)
            output = array_ops.concat([cell_output] + attentions, 1) @ W_out + B_out
            output_ta = output_ta.write(size, output)
            state_ta = state_ta.write(size, state)
            for i, (attn, weight) in enumerate(zip(attentions, weights)):
                attentions_ta[i] = attentions_ta[i].write(size, attn)
                weights_ta[i] = weights_ta[i].write(size, weight)
            return size + 1, state_ta, output_ta, attentions_ta, weights_ta

        def time_step(time, size, state_ta, output_ta, attentions_ta, weights_ta):
            arrays = state_ta, output_ta, attentions_ta, weights_ta
            with vs.variable_scope("LeftTimeStep"):
                size, *arrays = sub_time_step(left_cell, time, size, *arrays)
            with vs.variable_scope("RightTimeStep"):
                size, *arrays = sub_time_step(right_cell, time, size, *arrays)
            state_ta, output_ta, attentions_ta, weights_ta = arrays
            return time + 1, size, state_ta, output_ta, attentions_ta, weights_ta

        def cond(time, size, state_ta, output_ta, attentions_ta, weights_ta):
            return time < time_steps

        time = array_ops.constant(0, dtypes.int32, name="time")
        size = array_ops.constant(0, dtypes.int32, name="size")
        if initial_state is None:
            initial_state = array_ops.zeros([batch_size, state_size], dtype, "initial_state")
        if initial_state_attention:
            initial_attentions, _ = attention(initial_state)
        else:
            batch_attn_size = array_ops.stack([batch_size, attention.input_size])
            initial_attentions = [array_ops.zeros(batch_attn_size, dtype) for _ in range(attention.length)]
            for attn in initial_attentions:  # Ensure the second shape of attention vectors is set.
                attn.set_shape([None, attention.input_size])
        arrays = (state_ta, output_ta, attentions_ta, weights_ta)
        size, *arrays = sub_time_step(right_cell, time, size, *arrays, True)
        _, _, *arrays = control_flow_ops.while_loop(cond, time_step, (time, size, *arrays), parallel_iterations=1)
        state_ta, output_ta, attentions_ta, weights_ta = arrays
        indices = array_ops.stack(math_ops.range(input_length))
        outputs = output_ta.gather(indices)
        states = state_ta.gather(indices)
        attentions = [attention_ta.gather(indices) for attention_ta in attentions_ta]
        weights = [weight_ta.gather(indices) for weight_ta in weights_ta]
    return outputs, states, attentions, weights


def stack_attention_tree_dynamic_rnn(cell, inputs, attention, output_size=None,
                                     stack_size=None, initial_states=None, loop_function=None, stack_function=None,
                                     initial_state_attention=True, dtype=None, scope=None, stack_scope=None):
    """

    :param cell:
    :param inputs:
    :param attention:
    :param output_size:
    :param stack_size:
    :param initial_states: 3D-Tensor with shape [stack_size x batch_size x state_size]
    :param loop_function:
    :param stack_function:
    :param initial_state_attention:
    :param dtype:
    :param scope:
    :param stack_scope:
    :return:
    """
    assert isinstance(cell, (list, tuple)) and len(cell) == 2
    assert inputs.get_shape()[1].value == attention.batch_size
    assert attention.state_size == cell[0].state_size
    assert attention.state_size == cell[1].state_size
    assert cell[0].output_size == cell[1].output_size
    assert stack_size is not None or initial_states is not None
    if stack_size is None:
        stack_size = array_ops.shape(initial_states)[0]
    assert initial_states is None or initial_states.get_shape()[2].value == attention.state_size
    input_size = inputs.get_shape()[2].value
    assert stack_function is not None or output_size is None or output_size == input_size
    output_size = input_size if stack_function is None else output_size
    input_length = inputs.get_shape()[0]
    batch_size = inputs.get_shape()[1].value
    if output_size is None:
        output_size = cell[0].output_size
    state_size = cell[0].state_size

    with vs.variable_scope(stack_scope or "StackAttentionTreeDynamicRnn", dtype=dtype) as stack_scope:
        dtype = stack_scope.dtype

        with vs.variable_scope("Arrays"):
            shape = [input_length.value, batch_size, state_size]
            state_ta = ta_ops.TensorArray(dtype, stack_size, None, None, "states", element_shape=shape)
            shape = [input_length.value, batch_size, output_size]
            output_ta = ta_ops.TensorArray(dtype, stack_size, None, None, "outputs", element_shape=shape)
            weights_ta = []
            for i in range(attention.length):
                shape = [input_length.value, batch_size, None]
                ta = ta_ops.TensorArray(dtype, stack_size, None, None, "weights_%d" % i, element_shape=shape)
                weights_ta.append(ta)
            attentions_ta = []
            for i in range(attention.length):
                shape = [input_length.value, batch_size, attention.input_size]
                ta = ta_ops.TensorArray(dtype, stack_size, None, None, "attentions_%d" % i, element_shape=shape)
                attentions_ta.append(ta)

        def time_step(time, inputs, state_ta, output_ta, attentions_ta, weights_ta):
            with vs.variable_scope("TimeStep"):
                initial_state = None
                if initial_states is not None:
                    initial_state = array_ops.gather(initial_states, time)
                outputs, states, attentions, weights = attention_tree_dynamic_rnn(
                    cell, inputs, attention, output_size, initial_state, loop_function,
                    initial_state_attention, dtype, scope)
                inputs = outputs
                if stack_function is not None:
                    inputs = stack_function(inputs, outputs, states, time)
                state_ta = state_ta.write(time, states)
                output_ta = output_ta.write(time, outputs)
                for i, (attn, weight) in enumerate(zip(attentions, weights)):
                    attentions_ta[i] = attentions_ta[i].write(time, attn)
                    weights_ta[i] = weights_ta[i].write(time, weight)
            return time + 1, inputs, state_ta, output_ta, attentions_ta, weights_ta

        def cond(time, size, state_ta, output_ta, attentions_ta, weights_ta):
            return time < stack_size

        time = array_ops.constant(0, dtypes.int32, name="time")
        arrays = (state_ta, output_ta, attentions_ta, weights_ta)
        _, _, *arrays = control_flow_ops.while_loop(cond, time_step, (time, inputs, *arrays))
        state_ta, output_ta, attentions_ta, weights_ta = arrays
        outputs = output_ta.stack()
        states = state_ta.stack()
        attentions = [attention_ta.stack() for attention_ta in attentions_ta]
        weights = [weight_ta.stack() for weight_ta in weights_ta]
    return outputs, states, attentions, weights


def attention_dynamic_rnn(cell, inputs, attention, output_size=None, initial_output=None,
                          initial_state=None, loop_function=None,
                          initial_state_attention=True, dtype=None, scope=None):
    batch_size = inputs.get_shape()[1].value
    assert batch_size == attention.batch_size
    # assert attention.state_size == cell.state_size
    input_size = inputs.get_shape()[2].value
    if output_size is None:
        output_size = cell.output_size
    state_size = cell.state_size
    output_length = array_ops.shape(inputs)[0]

    with vs.variable_scope(scope or "AttentionDynamicRnn", dtype=dtype) as scope:
        dtype = scope.dtype

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("AttentionInputProjection"):
            W_inp = vs.get_variable(_WEIGHTS_NAME, [input_size + attention.output_size, input_size], dtype)
            B_inp = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)
        # with vs.variable_scope("AttentionOutputProjection"):
        #     W_out = vs.get_variable(_WEIGHTS_NAME, [cell.output_size + attention.output_size, output_size], dtype)
        #     B_out = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)

        with vs.variable_scope("Arrays"):
            shape = [batch_size, state_size]
            state_ta = ta_ops.TensorArray(dtype, output_length, None, None, "states", element_shape=shape)
            shape = [batch_size, output_size]
            output_ta = ta_ops.TensorArray(dtype, output_length, None, None, "outputs", element_shape=shape)
            weights_ta = []
            for i in range(attention.length):
                shape = [batch_size, None]
                ta = ta_ops.TensorArray(dtype, output_length, None, None, "weights_%d" % i, element_shape=shape)
                weights_ta.append(ta)
            attentions_ta = []
            for i in range(attention.length):
                shape = [batch_size, attention.input_size]
                ta = ta_ops.TensorArray(dtype, output_length, None, None, "attentions_%d" % i, element_shape=shape)
                attentions_ta.append(ta)

        def time_step(time, output, state, attentions, state_ta, output_ta, attentions_ta, weights_ta):
            with vs.variable_scope("Time-Step"):
                input = array_ops.gather(inputs, time)
                if loop_function is not None:
                    input = loop_function(input, output, state, time)
                # If loop_function is set, we use it instead of decoder_inputs.
                x = array_ops.concat([input] + attentions, 1) @ W_inp + B_inp
                # Run the RNN.
                cell_output, state = cell(x, state)
                # Run the attention mechanism.
                attentions, weights = attention(state)
                # output = array_ops.concat([cell_output] + attentions, 1) @ W_out + B_out
                output = cell_output
                state_ta = state_ta.write(time, state)
                output_ta = output_ta.write(time, output)
                for i, (attn, weight) in enumerate(zip(attentions, weights)):
                    attentions_ta[i] = attentions_ta[i].write(time, attn)
                    weights_ta[i] = weights_ta[i].write(time, weight)
            return time + 1, output, state, attentions, state_ta, output_ta, attentions_ta, weights_ta

        def cond(time, output, state, attentions, state_ta, output_ta, attentions_ta, weights_ta):
            return time < output_length

        if initial_state is None:
            state = array_ops.zeros([batch_size, cell.state_size], dtype, "initial_state")
        else:
            state = initial_state
        time = array_ops.constant(0, dtypes.int32, name="time")
        if initial_state_attention:
            attentions, weights = attention(state)
        else:
            batch_attn_size = array_ops.stack([batch_size, attention.input_size])
            attentions = [array_ops.zeros(batch_attn_size, dtype) for _ in range(attention.length)]
            for attn in attentions:  # Ensure the second shape of attention vectors is set.
                attn.set_shape([None, attention.input_size])
        if initial_output is None:
            output = array_ops.zeros([batch_size, output_size], dtype, "initial_output")
        else:
            output = initial_output

        arrays = (state_ta, output_ta, attentions_ta, weights_ta)
        _, _, _, _, *arrays = control_flow_ops.while_loop(cond, time_step, (time, output, state, attentions, *arrays))
        state_ta, output_ta, attentions_ta, weights_ta = arrays
        outputs = output_ta.stack()
        states = state_ta.stack()
        attentions = [attention_ta.stack() for attention_ta in attentions_ta]
        weights = [weight_ta.stack() for weight_ta in weights_ta]
    return outputs, states, attentions, weights


def stack_attention_dynamic_rnn(cell, inputs, attention, output_size=None, initial_outputs=None,
                                stack_size=None, initial_states=None, loop_function=None, stack_function=None,
                                initial_state_attention=True, dtype=None, scope=None, stack_scope=None):
    batch_size = inputs.get_shape()[1].value
    assert attention.batch_size == batch_size
    state_size = cell.state_size
    assert attention.state_size == state_size
    assert initial_states is None or initial_states.get_shape()[2].value == state_size
    assert stack_size is not None or initial_states is not None
    if stack_size is None:
        stack_size = array_ops.shape(initial_states)[0]
    input_size = inputs.get_shape()[2].value
    assert stack_function is not None or output_size is None or output_size == input_size
    output_size = input_size if stack_function is None else output_size
    assert initial_outputs is None or output_size is None or initial_outputs.get_shape()[2].value == output_size
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(stack_scope or "StackAttentionDynamicRnn", dtype=dtype) as stack_scope:
        dtype = stack_scope.dtype

        with vs.variable_scope("Arrays"):
            shape = [None, batch_size, state_size]
            state_ta = ta_ops.TensorArray(dtype, stack_size, None, None, "states", element_shape=shape)
            shape = [None, batch_size, output_size]
            output_ta = ta_ops.TensorArray(dtype, stack_size, None, None, "outputs", element_shape=shape)
            weights_ta = []
            for i in range(attention.length):
                shape = [None, batch_size, None]
                ta = ta_ops.TensorArray(dtype, stack_size, None, None, "weights_%d" % i, element_shape=shape)
                weights_ta.append(ta)
            attentions_ta = []
            for i in range(attention.length):
                shape = [None, batch_size, attention.input_size]
                ta = ta_ops.TensorArray(dtype, stack_size, None, None, "attentions_%d" % i, element_shape=shape)
                attentions_ta.append(ta)

        def time_step(time, inputs, state_ta, output_ta, attentions_ta, weights_ta):
            with vs.variable_scope("TimeStep"):
                initial_state = None
                if initial_states is not None:
                    initial_state = array_ops.gather(initial_states, time)
                initial_output = None
                if initial_output is not None:
                    initial_output = array_ops.gather(initial_outputs, time)
                outputs, states, attentions, weights = attention_dynamic_rnn(
                    cell, inputs, attention, output_size, initial_output, initial_state, loop_function,
                    initial_state_attention, dtype, scope)
                inputs = outputs
                if stack_function is not None:
                    inputs = stack_function(inputs, outputs, states, time)
                state_ta = state_ta.write(time, states)
                output_ta = output_ta.write(time, outputs)
                for i, (attn, weight) in enumerate(zip(attentions, weights)):
                    attentions_ta[i] = attentions_ta[i].write(time, attn)
                    weights_ta[i] = weights_ta[i].write(time, weight)
            return time + 1, inputs, state_ta, output_ta, attentions_ta, weights_ta

        def cond(time, inputs, state_ta, output_ta, attentions_ta, weights_ta):
            return time < stack_size

        time = array_ops.constant(0, dtypes.int32, name="time")
        arrays = (state_ta, output_ta, attentions_ta, weights_ta)
        _, _, *arrays = control_flow_ops.while_loop(cond, time_step, (time, inputs, *arrays))
        state_ta, output_ta, attentions_ta, weights_ta = arrays
        outputs = output_ta.stack()
        states = state_ta.stack()
        attentions = [attention_ta.stack() for attention_ta in attentions_ta]
        weights = [weight_ta.stack() for weight_ta in weights_ta]
    return outputs, states, attentions, weights
