from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

# noinspection PyProtectedMember
_concat = rnn_cell_impl._concat
# noinspection PyProtectedMember
_infer_state_dtype = rnn._infer_state_dtype
# noinspection PyProtectedMember
_reverse_seq = rnn._reverse_seq
# noinspection PyProtectedMember
_rnn_step = rnn._rnn_step
# noinspection PyProtectedMember
_linear = rnn_cell_impl._linear


def static_rnn(cell, inputs, initial_state=None, dtype=None,
               sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:

      ```python
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
          output, state = cell(input_, state)
          outputs.append(output)
        return (outputs, state)
      ```
      However, a few other options are available:

      An initial state can be provided.
      If the sequence_length vector is provided, dynamic calculation is performed.
      This method of calculation does not compute the RNN steps past the maximum
      sequence length of the minibatch (thus saving computational time),
      and properly propagates the state at an example's sequence length
      to the final state output.

      The dynamic calculation performed is, at time `t` for batch row `b`,

      ```python
        (output, state)(b, t) =
          (t >= sequence_length(b))
            ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
            : cell(input(b, t), state(b, t - 1))
      ```

      Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a `Tensor` of shape
          `[batch_size, input_size]`, or a nested tuple of such elements.
        initial_state: (optional) An initial state for the RNN.
          If `cell.state_size` is an integer, this must be
          a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
          If `cell.state_size` is a tuple, this should be a tuple of
          tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state and expected output.
          Required if initial_state is not provided or RNN state has a heterogeneous
          dtype.
        sequence_length: Specifies the length of each sequence in inputs.
          An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "rnn".

      Returns:
        A pair (outputs, state) where:

        - outputs is a length T list of outputs (one for each input), or a nested
          tuple of such elements.
        - state is the final state

      Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
          (column size) cannot be inferred from inputs via shape inference.
      """

    if not isinstance(cell, core_rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while nest.is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare configurations
            sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size")

            def _create_zero_output(_output_size):
                # convert int to TensorShape if necessary
                _size = _concat(_output_size, prefix=[batch_size])
                output = array_ops.zeros(
                    array_ops.stack(_size), _infer_state_dtype(dtype, state))
                shape = _concat(
                    _output_size, prefix=[fixed_batch_size.value])
                output.set_shape(tensor_shape.TensorShape(shape))
                return output

            output_size = cell.output_size
            flat_output_size = nest.flatten(output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = nest.pack_sequence_as(structure=output_size,
                                                flat_sequence=flat_zero_output)

            sequence_length = math_ops.to_int32(sequence_length)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            call_cell = lambda: cell(input_, state)
            if sequence_length is not None:
                # noinspection PyUnboundLocalVariable
                (_, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (_, state) = call_cell()

            states.append(state)

    return states


def static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                             initial_state_fw=None, initial_state_bw=None,
                             dtype=None, sequence_length=None, scope=None):
    """Creates a bidirectional recurrent neural network.

    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.

    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size], or a nested tuple of such elements.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"

    Returns:
      A tuple (outputs, states_fw, states_bw) where:
        outputs is a length `T` list of outputs (one for each input), which
          are depth-concatenated forward and backward outputs.
        states_fw is the final state of the forward rnn.
        states_bw is the final state of the backward rnn.

    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell_fw, core_rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, core_rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            states_fw = static_rnn(
                cell_fw, inputs, initial_state_fw, dtype,
                sequence_length, scope=fw_scope)

        # Backward direction
        with vs.variable_scope("bw") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            states_bw = static_rnn(
                cell_bw, reversed_inputs, initial_state_bw,
                dtype, sequence_length, scope=bw_scope)

    return states_fw, states_bw


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.
  
    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.
  
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.
  
    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
  
    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = vs.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                vs.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for head_index in range(num_heads):
                with vs.variable_scope("Attention_%d" % head_index):
                    y = _linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[head_index] * math_ops.tanh(hidden_features[head_index] + y),
                                            [2, 3])
                    attn = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(attn, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        states = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                vs.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with vs.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = _linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            states.append(state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with vs.variable_scope(
                        vs.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with vs.variable_scope("AttnOutputProjection"):
                output = _linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, states
