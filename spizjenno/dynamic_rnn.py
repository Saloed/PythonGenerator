from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, rnn
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

# noinspection PyProtectedMember
_concat = rnn_cell_impl._concat
# noinspection PyProtectedMember
_rnn_step = rnn._rnn_step

_BIAS_NAME = "biases"
_WEIGHTS_NAME = "weights"


def infer_state_dtype(explicit_dtype, state):
    """Infer the dtype of an RNN state.

    Args:
      explicit_dtype: explicitly declared dtype or None.
      state: RNN's hidden state. Must be a Tensor or a nested iterable containing
        Tensors.

    Returns:
      dtype: inferred dtype of hidden state.

    Raises:
      ValueError: if `state` has heterogeneous dtypes or is empty.
    """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to infer a "
                "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    """Creates a dynamic version of bidirectional recurrent neural network.

    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs. The input_size of forward and
    backward cell must match. The initial state for both directions is zero by
    default (but can be set optionally) and no intermediate states are ever
    returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not
    given.

    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: The RNN inputs.
        If time_major == False (default), this must be a tensor of shape:
          `[batch_size, max_time, input_size]`.
        If time_major == True, this must be a tensor of shape:
          `[max_time, batch_size, input_size]`.
        [batch_size, input_size].
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences in the batch.
        If not provided, all batch entries are assumed to be full sequences; and
        time reversal is applied from time `0` to `max_time` for each sequence.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial states and expected output.
        Required if initial_states are not provided or RNN states have a
        heterogeneous dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"

    Returns:
      A tuple (outputs, output_states) where:
        outputs: A tuple (output_fw, output_bw) containing the forward and
          the backward rnn output `Tensor`.
          If time_major == False (default),
            output_fw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_bw.output_size]`.
          If time_major == True,
            output_fw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_bw.output_size]`.
          It returns a tuple instead of a single concatenated `Tensor`, unlike
          in the `bidirectional_rnn`. If the concatenated one is preferred,
          the forward and backward outputs can be concatenated as
          `tf.concat(outputs, 2)`.
        output_states: A tuple (output_state_fw, output_state_bw) containing
          the forward and the backward final states of bidirectional rnn.

    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    """

    # pylint: disable=protected-access
    if not isinstance(cell_fw, rnn_cell_impl.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, rnn_cell_impl.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    # pylint: enable=protected-access

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_states_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            if seq_lengths is not None:
                return array_ops.reverse_sequence(
                    input=input_, seq_lengths=seq_lengths,
                    seq_dim=seq_dim, batch_dim=batch_dim)
            else:
                return array_ops.reverse(input_, axis=[seq_dim])

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            output_bw, output_states_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=bw_scope)

        output_bw = _reverse(
            output_bw, seq_lengths=sequence_length,
            seq_dim=time_dim, batch_dim=batch_dim)

        output_states_bw = _reverse(
            output_states_bw, seq_lengths=sequence_length,
            seq_dim=time_dim, batch_dim=batch_dim)

        outputs = (output_fw, output_bw)
        output_states = (output_states_fw, output_states_bw)

    return outputs, output_states


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    This function is functionally identical to the function `rnn` above, but
    performs fully dynamic unrolling of `inputs`.

    Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`, one for
    each frame.  Instead, `inputs` may be a single `Tensor` where
    the maximum time is either the first or second dimension (see the parameter
    `time_major`).  Alternatively, it may be a (possibly nested) tuple of
    Tensors, each of them having matching batch and time dimensions.
    The corresponding output is either a single `Tensor` having the same number
    of time steps and batch size, or a (possibly nested) tuple of such tensors,
    matching the nested structure of `cell.output_size`.

    The parameter `sequence_length` is optional and is used to copy-through state
    and zero-out outputs when past a batch element's sequence length. So it's more
    for correctness than performance, unlike in rnn().

    Args:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.

        If `time_major == False` (default), this must be a `Tensor` of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such
          elements.

        If `time_major == True`, this must be a `Tensor` of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such
          elements.

        This may also be a (possibly nested) tuple of Tensors satisfying
        this property.  The first two dimensions must match across all the inputs,
        but otherwise the ranks and other shape components may differ.
        In this case, input to `cell` at each time-step will replicate the
        structure of these tuples, except for the time dimension (from which the
        time is taken).

        The input to `cell` at each time step will be a `Tensor` or (possibly
        nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
      sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
      A pair (outputs, state) where:

        outputs: The RNN output `Tensor`.

          If time_major == False (default), this will be a `Tensor` shaped:
            `[batch_size, max_time, cell.output_size]`.

          If time_major == True, this will be a `Tensor` shaped:
            `[max_time, batch_size, cell.output_size]`.

          Note, if `cell.output_size` is a (possibly nested) tuple of integers
          or `TensorShape` objects, then `outputs` will be a tuple having the
          same structure as `cell.output_size`, containing Tensors having shapes
          corresponding to the shape data in `cell.output_size`.

        state: The final state.  If `cell.state_size` is an int, this
          will be shaped `[batch_size, cell.state_size]`.  If it is a
          `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
          If it is a (possibly nested) tuple of ints or `TensorShape`, this will
          be a tuple having the corresponding shapes.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    # pylint: disable=protected-access
    if not isinstance(cell, rnn_cell_impl.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    # pylint: enable=protected-access

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
        # (B,T,D) => (T,B,D)
        flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                           for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.get_shape())
        sequence_length = array_ops.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
        batch_size = input_shape[0][1]

        for input_ in input_shape:
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError("All inputs should have the same batch size")

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

        (outputs, final_states) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            flat_output = nest.flatten(outputs)
            flat_output = [array_ops.transpose(output, [1, 0, 2])
                           for output in flat_output]
            outputs = nest.pack_sequence_as(
                structure=outputs, flat_sequence=flat_output)

            flat_states = nest.flatten(final_states)
            flat_states = [array_ops.transpose(state, [1, 0, 2])
                           for state in flat_states]
            final_states = nest.pack_sequence_as(
                structure=final_states, flat_sequence=flat_states)

    return outputs, final_states


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
    """Internal implementation of Dynamic RNN.

    Args:
      cell: An instance of RNNCell.
      inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
        tuple of such elements.
      initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
        `cell.state_size` is a tuple, then this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      parallel_iterations: Positive Python int.
      swap_memory: A Python boolean
      sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
      dtype: (optional) Expected dtype of output. If not specified, inferred from
        initial_state.

    Returns:
      Tuple `(final_outputs, final_state)`.
      final_outputs:
        A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
        `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
        objects, then this returns a (possibly nsted) tuple of Tensors matching
        the corresponding shapes.
      final_state:
        A `Tensor`, or possibly nested tuple of Tensors, matching in length
        and shapes to `initial_state`.

    Raises:
      ValueError: If the input depth cannot be inferred via shape inference
        from the inputs.
    """
    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_state_size = nest.flatten(cell.state_size)
    flat_output_size = nest.flatten(cell.output_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(
            array_ops.stack(size), infer_state_dtype(dtype, state))

    flat_zero_output = tuple(create_zero_arrays(output) for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def create_ta(name, _dtype):
        return tensor_array_ops.TensorArray(dtype=_dtype, size=time_steps, tensor_array_name=base_name + name)

    state_ta = tuple(create_ta("state_%d" % i, infer_state_dtype(dtype, state)) for i in range(len(flat_state_size)))
    output_ta = tuple(create_ta("output_%d" % i, infer_state_dtype(dtype, state)) for i in range(len(flat_output_size)))
    input_ta = tuple(create_ta("input_%d" % i, flat_input[0].dtype) for i in range(len(flat_input)))
    input_ta = tuple(ta.unstack(input) for ta, input in zip(input_ta, flat_input))

    def _time_step(_time, output_ta_t, _state, state_ta_t):
        """Take a time step of the dynamic RNN.

        Args:
          _time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          _state: nested tuple of vector tensors that represent the state.

        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        input_t = tuple(ta.read(_time) for ta in input_ta)
        # Restore some shape information
        for input, _shape in zip(input_t, inputs_got_shape):
            input.set_shape(_shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=_time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=_state,
                call_cell=lambda: cell(input_t, _state),
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = cell(input_t, _state)

        # Pack state if using state tuples
        output = nest.flatten(output)
        output_ta_t = tuple(ta.write(_time, out) for ta, out in zip(output_ta_t, output))

        # Pack state if using state tuples
        flatten_state = nest.flatten(new_state)
        state_ta_t = tuple(ta.write(_time, stt) for ta, stt in zip(state_ta_t, flatten_state))

        return _time + 1, output_ta_t, new_state, state_ta_t

    _, output_final_ta, _, state_final_ta = control_flow_ops.while_loop(
        cond=lambda _time, *_: _time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state, state_ta),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _concat([const_time_steps, const_batch_size], output_size, True)
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=final_outputs)

    # Unpack final output if not using state tuples.
    final_states = tuple(ta.stack() for ta in state_final_ta)

    # Restore some shape information
    for state, state_size in zip(final_states, flat_output_size):
        shape = _concat([const_time_steps, const_batch_size], state_size, True)
        state.set_shape(shape)

    final_states = nest.pack_sequence_as(structure=cell.state_size, flat_sequence=final_states)

    return final_outputs, final_states


def attention_dynamic_rnn(
        cell,
        inputs,
        attention_states,
        output_size=None,
        initial_state=None,
        num_heads=1,
        loop_function=None,
        dtype=None,
        scope=None,
        initial_state_attention=False
):
    """RNN decoder with attention for the sequence-to-sequence model.

    Args:
      inputs: A 3D Tensors [sequence_length x batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: list of 3D Tensors or 3D Tensor [batch_size x attn_length x attn_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate
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
          It is a 2D Tensor of shape [sequence_length x batch_size x cell.state_size].

    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
    """
    if not isinstance(attention_states, list):
        attention_states = [attention_states]
    if len(attention_states) == 0:
        raise ValueError("Number of attention states must be greater zero")
    if inputs.get_shape()[0].value == 0:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    for attn in attention_states:
        if attn.get_shape()[2].value is None:
            raise ValueError("Shape[2] of attention_states must be known: %s" % attn.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(scope or "attention_dynamic_rnn", dtype=dtype) as scope:
        dtype = scope.dtype

        time_steps = array_ops.shape(inputs)[0]

        batch_size = inputs.get_shape()[1].value
        input_size = inputs.get_shape()[2].value

        num_attentions = len(attention_states)
        attn_size = attention_states[0].get_shape()[2].value

        hiddens_features = []
        vectors = []
        hiddens = []
        attn_lengths = []
        for attn_index, attn in enumerate(attention_states):
            with vs.variable_scope("AttentionHidden_%d" % attn_index):
                attn_length = attn.get_shape()[1].value
                if attn_length is None:
                    attn_length = array_ops.shape(attn)[1]

                # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
                hidden = array_ops.reshape(attn, [batch_size, attn_length, 1, attn_size])

                hidden_features = []
                vector = []
                for head_index in range(num_heads):
                    k = vs.get_variable("AttnW_%d" % head_index, [1, 1, attn_size, attn_size])
                    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                    vector.append(vs.get_variable("AttnV_%d" % head_index, [attn_size]))

            hiddens_features.append(hidden_features)
            vectors.append(vector)
            hiddens.append(hidden)
            attn_lengths.append(attn_length)

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("AttentionInputProjection"):
            input_length = input_size + attn_size * num_heads * num_attentions
            W_inp = vs.get_variable(_WEIGHTS_NAME, [input_length, input_size], dtype)
            B_inp = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)
        with vs.variable_scope("AttentionOutputProjection"):
            output_length = cell.output_size + attn_size * num_heads * num_attentions
            W_out = vs.get_variable(_WEIGHTS_NAME, [output_length, output_size], dtype)
            B_out = vs.get_variable(_BIAS_NAME, [output_size], dtype, bias_initializer)
        with vs.variable_scope("AttentionStateProjection"):
            W_stt = vs.get_variable(_WEIGHTS_NAME, [cell.state_size, attn_size], dtype)
            B_stt = vs.get_variable(_BIAS_NAME, [attn_size], dtype, bias_initializer)

        with vs.variable_scope("Arrays"):
            name = "outputs"
            shape = [batch_size, output_size]
            output_ta = tensor_array_ops.TensorArray(dtype, time_steps, tensor_array_name=name, element_shape=shape)
            name = "states"
            shape = [batch_size, cell.state_size]
            state_ta = tensor_array_ops.TensorArray(dtype, time_steps, tensor_array_name=name, element_shape=shape)
            attentions_weights_ta = []
            for i in range(num_attentions * num_heads):
                ta = tensor_array_ops.TensorArray(dtype, time_steps, tensor_array_name="attentions_weights_%d" % i)
                attentions_weights_ta.append(ta)

        def attention(_state):
            """Put attention masks on hidden using hidden_features and query."""
            with vs.variable_scope("Attention"):
                attentions = []  # Results of attention reads will be stored here.
                weights = []
                for i in range(num_attentions):
                    attn_length = attn_lengths[i]
                    hidden = hiddens[i]
                    for j in range(num_heads):
                        vector = vectors[i][j]
                        hidden_features = hiddens_features[i][j]
                        y = _state @ W_stt + B_stt
                        y = array_ops.reshape(y, [batch_size, 1, 1, attn_size])
                        # Attention mask is a softmax of v' * tanh(...).
                        s = math_ops.reduce_sum(vector * math_ops.tanh(hidden_features + y), [2, 3])
                        weight = math_ops.sigmoid(s)
                        weights.append(weight)
                        weight = array_ops.reshape(weight, [batch_size, attn_length, 1, 1])
                        # Now calculate the attention-weighted vector.
                        attention = math_ops.reduce_sum(weight * hidden, [1, 2])
                        attention = array_ops.reshape(attention, [batch_size, attn_size])
                        attentions.append(attention)
            return attentions, weights

        def time_step(time, output_ta, state_ta, attentions_weights_ta, input, state, attentions):
            with vs.variable_scope("Time-Step"):
                if loop_function is None:
                    input = array_ops.gather(inputs, time)
                # If loop_function is set, we use it instead of decoder_inputs.
                x = array_ops.concat([input] + attentions, 1) @ W_inp + B_inp
                # Run the RNN.
                cell_output, state = cell(x, state)
                # Run the attention mechanism.
                attentions, attentions_weights = attention(state)
                output = array_ops.concat([cell_output] + attentions, 1) @ W_out + B_out
                if loop_function is not None:
                    input = loop_function(output, time)
                output_ta = output_ta.write(time, output)
                state_ta = state_ta.write(time, state)
                for i in range(num_heads * num_attentions):
                    attentions_weights_ta[i] = attentions_weights_ta[i].write(time, attentions_weights[i])
            return time + 1, output_ta, state_ta, attentions_weights_ta, input, state, attentions

        if initial_state is None:
            initial_state = array_ops.zeros([batch_size, cell.state_size], dtype, "initial_state")

        batch_attn_size = array_ops.stack([batch_size, attn_size])
        if initial_state_attention:
            attentions = attention(initial_state)
        else:
            attentions = [array_ops.zeros(batch_attn_size, dtype) for _ in range(num_heads * num_attentions)]
            for attn in attentions:  # Ensure the second shape of attention vectors is set.
                attn.set_shape([None, attn_size])

        time = array_ops.constant(0, dtypes.int32, name="time")
        input = array_ops.gather(inputs, 0)

        _, output_ta, state_ta, attentions_weights_ta, *_ = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=time_step,
            loop_vars=(time, output_ta, state_ta, attentions_weights_ta, input, initial_state, attentions)
        )

        outputs = output_ta.stack()
        states = state_ta.stack()
        attentions_weights = [attentions_weights_ta[i].stack() for i in range(num_attentions * num_heads)]
    return outputs, states, attentions_weights


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    dtype=None,
                                    scope=None):
    """Creates a dynamic bidirectional recurrent neural network.
    Stacks several bidirectional rnn layers. The combined forward and backward
    layer outputs are used as input of the next layer. tf.bidirectional_rnn
    does not allow to share forward and backward information between layers.
    The input_size of the first forward and backward cells must match.
    The initial state for both directions is zero and no intermediate states
    are returned.
    Args:
      cells_fw: List of instances of RNNCell, one per layer,
        to be used for forward direction.
      cells_bw: List of instances of RNNCell, one per layer,
        to be used for backward direction.
      inputs: The RNN inputs. this must be a tensor of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      initial_states_fw: (optional) A list of the initial states (one per layer)
        for the forward RNN.
        Each tensor must has an appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
      initial_states_bw: (optional) Same as for `initial_states_fw`, but using
        the corresponding properties of `cells_bw`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      scope: VariableScope for the created subgraph; defaults to None.
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs: Output `Tensor` shaped:
          `batch_size, max_time, layers_output]`. Where layers_output
          are depth-concatenated forward and backward outputs.
        output_states_fw is the final states, one tensor per layer,
          of the forward rnn.
        output_states_bw is the final states, one tensor per layer,
          of the backward rnn.
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is `None`.
    """
    if not cells_fw:
        raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if initial_states_fw is not None and (not isinstance(cells_fw, list) or len(cells_fw) != len(cells_fw)):
        raise ValueError("initial_states_fw must be a list of state tensors (one per layer).")
    if initial_states_bw is not None and (not isinstance(cells_bw, list) or len(cells_bw) != len(cells_bw)):
        raise ValueError("initial_states_bw must be a list of state tensors (one per layer).")

    res_states_fw = []
    res_states_bw = []
    prev_outputs = inputs

    with vs.variable_scope(scope or "stack_bidirectional_dynamic_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]
            with vs.variable_scope("cell_%d" % i):
                outputs, (states_fw, states_bw) = bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_outputs,
                    sequence_length,
                    initial_state_fw,
                    initial_state_bw,
                    dtype,
                    parallel_iterations)
                # Concat the outputs to create the new input.
                prev_outputs = array_ops.concat(outputs, 2)
            res_states_fw.append(states_fw)
            res_states_bw.append(states_bw)

    return prev_outputs, tuple(res_states_fw), tuple(res_states_bw)


def stack_attention_dynamic_rnn(cells,
                                inputs,
                                attention_states,
                                output_size=None,
                                num_heads=1,
                                loop_function=None,
                                use_inputs=False,
                                initial_states=None,
                                dtype=None,
                                scope=None):
    """Creates a dynamic attention recurrent neural network.

    Args:
      cells: List of instances of RNNCell, one per layer,
        to be used for forward direction.
      inputs: A 3D Tensors [sequence_length x batch_size x input_size].
      attention_states: ...
      output_size: ...
      num_heads: ...
      loop_function: ...
      use_inputs: ...
      initial_states: (optional) A list of the initial states (one per layer)
        for the forward RNN.
        Each tensor must has an appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      scope: VariableScope for the created subgraph; defaults to None.
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs: Output `Tensor` shaped:
          `batch_size, max_time, layers_output]`. Where layers_output
          are depth-concatenated forward and backward outputs.
        output_states is the final states, one tensor per layer.
    Raises:
      TypeError: If `cell` is not an instance of `RNNCell`.
      ValueError: If inputs is `None`.
    """
    if not cells:
        raise ValueError("Must specify at least one fw cell for RNN.")
    if not isinstance(cells, list):
        raise ValueError("cells must be a list of RNNCells (one per layer).")
    if initial_states is not None and not isinstance(cells, list):
        raise ValueError("initial_states must be a list of state tensors (one per layer).")
    if initial_states is not None and len(initial_states) != len(cells):
        raise ValueError("initial_states must be a list of state tensors (one per layer).")
    if not loop_function and not use_inputs:
        input_size = inputs.get_shape()[2].value

        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("LoopFunctionVariables"):
            W_loop = vs.get_variable(_WEIGHTS_NAME, [output_size, input_size], dtype)
            B_loop = vs.get_variable(_BIAS_NAME, [input_size], dtype, bias_initializer)

        def loop_function(prev, _):
            return nn_ops.relu(prev @ W_loop + B_loop)

    with vs.variable_scope(scope or "stack_attention_dynamic_rnn"):
        res_outputs = []
        res_states = []
        res_weighs = []
        outputs = inputs
        for i, cell in enumerate(cells):
            initial_state = None
            if initial_states:
                initial_state = initial_states[i]
            with vs.variable_scope("cell_%d" % i):
                outputs, states, weighs = attention_dynamic_rnn(
                    cell,
                    outputs,
                    attention_states,
                    output_size,
                    initial_state,
                    num_heads,
                    loop_function,
                    dtype)
            loop_function = None
            res_outputs.append(outputs)
            res_states.append(states)
            res_weighs.append(weighs)

    return tuple(res_outputs), tuple(res_states), tuple(res_weighs)
