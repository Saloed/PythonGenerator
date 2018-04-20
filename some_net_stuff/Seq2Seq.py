import copy
from collections import namedtuple

import numpy as np
import tensorflow as tf
from seq2seq import losses as seq2seq_losses
from seq2seq.configurable import Configurable
from seq2seq.contrib.seq2seq import helper
from seq2seq.contrib.seq2seq.helper import CustomHelper
from seq2seq.graph_module import GraphModule
from seq2seq.training import utils as training_utils
from seq2seq.training.utils import cell_from_spec
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell
from tensorflow.python.util import nest
from tensorflow.python.ops import embedding_ops


def _unpack_cell(cell):
    """Unpack the cells because the stack_bidirectional_dynamic_rnn
    expects a list of cells, one per layer."""
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
        return cell._cells  # pylint: disable=W0212
    else:
        return [cell]


def _transpose_batch_time(x):
    """Transpose the batch and time dimensions of a Tensor.

    Retains as much of the static shape information as possible.

    Args:
      x: A tensor of rank 2 or higher.

    Returns:
      x transposed along the first two dimensions.

    Raises:
      ValueError: if `x` is rank 1 or lower.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(
        x, array_ops.concat(
            ([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tensor_shape.TensorShape([
            x_static_shape[1].value, x_static_shape[0].value
        ]).concatenate(x_static_shape[2:]))
    return x_t


def _default_rnn_cell_params():
    """Creates default parameters used by multiple RNN encoders.
    """
    return {
        "cell_class": "GRUCell",
        "cell_params": {
            "num_units": 128,
            # "state_is_tuple": False,
        },
        "dropout_input_keep_prob": 1.0,
        "dropout_output_keep_prob": 1.0,
        "num_layers": 1,
        "residual_connections": False,
        "residual_combiner": "add",
        "residual_dense": False
    }


def _toggle_dropout(cell_params, mode):
    """Disables dropout during eval/inference mode
    """
    cell_params = copy.deepcopy(cell_params)
    if mode != ModeKeys.TRAIN:
        cell_params["dropout_input_keep_prob"] = 1.0
        cell_params["dropout_output_keep_prob"] = 1.0
    return cell_params


EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class Encoder(GraphModule, Configurable):
    """
     A bidirectional RNN encoder. Uses the same cell for both the
     forward and backward RNN. Stacking should be performed as part of
     the cell.

     Args:
       cell: An instance of tf.contrib.rnn.RNNCell
       name: A name for the encoder
     """

    def __init__(self, params, mode, name):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    def _build(self, inputs, *args, **kwargs):
        return self.encode(inputs, *args, **kwargs)

    def encode(self, inputs, sequence_length, **kwargs):
        """
        Encodes an input sequence.

        Args:
          inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
          sequence_length: The length of each input. An int32 tensor of shape [T].

        Returns:
          An `EncoderOutput` tuple containing the outputs and final state.
        """

        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

        # Concatenate outputs and states of the forward and backward RNNs
        outputs_concat = tf.concat(outputs, 2)

        return EncoderOutput(
            outputs=outputs_concat,
            final_state=states,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }


class AttentionDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context"
    ])):
    """Augmented decoder output that also includes the attention scores.
    """
    pass


class AttentionLayerDot(GraphModule, Configurable):
    """
    Attention layer according to https://arxiv.org/abs/1409.0473.

    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, params, mode, name="attention"):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    @staticmethod
    def default_params():
        return {"num_units": 128}

    def score_fn(self, keys, query):
        """Calculates a batch- and timweise dot product"""
        return tf.reduce_sum(keys * tf.expand_dims(query, 0), [2], name="att_sum_dot")

    def _build(self, query, keys, values, values_length):
        """Computes attention scores and outputs.

        Args:
          query: The query used to calculate attention scores.
            In seq2seq this is typically the current state of the decoder.
            A tensor of shape `[B, ...]`
          keys: The keys used to calculate attention scores. In seq2seq, these
            are typically the outputs of the encoder and equivalent to `values`.
            A tensor of shape `[B, T, ...]` where each element in the `T`
            dimension corresponds to the key for that value.
          values: The elements to compute attention over. In seq2seq, this is
            typically the sequence of encoder outputs.
            A tensor of shape `[B, T, input_dim]`.
          values_length: An int32 tensor of shape `[B]` defining the sequence
            length of the attention values.

        Returns:
          A tuple `(scores, context)`.
          `scores` is vector of length `T` where each element is the
          normalized "score" of the corresponding `inputs` element.
          `context` is the final attention layer output corresponding to
          the weighted inputs.
          A tensor fo shape `[B, input_dim]`.
        """
        values_depth = values.get_shape().as_list()[-1]

        # Fully connected layers to transform both keys and query
        # into a tensor with `num_units` units
        att_keys = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=self.params["num_units"],
            activation_fn=None,
            scope="att_keys")
        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=self.params["num_units"],
            activation_fn=None,
            scope="att_query")

        scores = self.score_fn(att_keys, att_query)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[0]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores_mask = tf.transpose(scores_mask, [1, 0])
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 0, name="context")
        # context.set_shape([None, values_depth])

        return scores_normalized, context


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""

    def _t(s):
        return (s if isinstance(s, ops.Tensor) else constant_op.constant(
            tensor_shape.TensorShape(s).as_list(),
            dtype=dtypes.int32,
            name="zero_suffix_shape"))

    def _create(s, d):
        return array_ops.zeros(
            array_ops.concat(
                ([batch_size], _t(s)), axis=0), dtype=d)

    return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    """Perform dynamic decoding with `decoder`.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if maximum_iterations is provided but is not a scalar.
    """
    if not isinstance(decoder, Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                        type(decoder))

    with variable_scope.variable_scope(scope or "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(
                maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tensor_shape.TensorShape):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: 1-D bool tensor.

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
            """
            (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs, state)
            next_finished = math_ops.logical_or(decoder_finished, finished)
            if maximum_iterations is not None:
                next_finished = math_ops.logical_or(
                    next_finished, time + 1 >= maximum_iterations)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return time + 1, outputs_ta, next_state, next_inputs, next_finished

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs,
                initial_finished
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state


class Decoder(GraphModule, Configurable):
    """An RNN Decoder that uses attention over an input sequence.

       Args:
         cell: An instance of ` tf.contrib.rnn.RNNCell`
         helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
         initial_state: A tensor or tuple of tensors used as the initial cell
           state.
         vocab_size: Output vocabulary size, i.e. number of units
           in the softmax layer
         attention_keys: The sequence used to calculate attention scores.
           A tensor of shape `[B, T, ...]`.
         attention_values: The sequence to attend over.
           A tensor of shape `[B, T, input_dim]`.
         attention_values_length: Sequence length of the attention values.
           An int32 Tensor of shape `[B]`.
         attention_fn: The attention function to use. This function map from
           `(state, inputs)` to `(attention_scores, attention_context)`.
           For an example, see `seq2seq.decoder.attention.AttentionLayer`.
         reverse_scores: Optional, an array of sequence length. If set,
           reverse the attention scores in the output. This is used for when
           a reversed source sequence is fed as an input but you want to
           return the scores in non-reversed order.
       """

    def __init__(self,
                 params,
                 mode,
                 vocab_size,
                 attention_keys,
                 attention_values,
                 attention_values_length,
                 attention_fn,
                 reverse_scores_lengths=None,
                 name="attention_decoder"):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)
        rnn_params = self.params["rnn_cell"]
        self.params["rnn_cell"] = _toggle_dropout(rnn_params, mode)
        dropout_input_keep_prob = rnn_params['dropout_input_keep_prob']
        dropout_output_keep_prob = rnn_params['dropout_output_keep_prob']
        cells = []
        for _ in range(rnn_params['num_layers']):
            cell = cell_from_spec(rnn_params['cell_class'], rnn_params['cell_params'])
            if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell=cell,
                    input_keep_prob=dropout_input_keep_prob,
                    output_keep_prob=dropout_output_keep_prob)
            cells.append(cell)

        if len(cells) > 1:
            final_cell = MultiRNNCell(
                cells=cells,
                state_is_tuple=False,
            )
        else:
            final_cell = cells[0]
        self.cell = final_cell
        # Not initialized yet
        self.initial_state = None
        self.helper = None

        self.vocab_size = vocab_size
        self.attention_keys = attention_keys
        self.attention_values = attention_values
        self.attention_values_length = attention_values_length
        self.attention_fn = attention_fn
        self.reverse_scores_lengths = reverse_scores_lengths

    def _setup(self, initial_state, helper):
        self.initial_state = initial_state

        def att_next_inputs(time, outputs, state, sample_ids, name=None):
            """Wraps the original decoder helper function to append the attention
            context.
            """
            finished, next_inputs, next_state = helper.next_inputs(
                time=time,
                outputs=outputs,
                state=state,
                sample_ids=sample_ids,
                name=name)
            next_inputs = tf.concat([next_inputs, outputs.attention_context], 1)
            return finished, next_inputs, next_state

        self.helper = CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=att_next_inputs)

    def finalize(self, outputs, final_state):
        """Applies final transformation to the decoder output once decoding is
        finished.
        """
        return outputs, final_state

    @staticmethod
    def default_params():
        return {
            "max_decode_length": 100,
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def _build(self, initial_state, helper):
        if not self.initial_state:
            self._setup(initial_state, helper)

        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        maximum_iterations = helper.max_time
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            maximum_iterations = self.params["max_decode_length"]

        outputs, final_state = dynamic_decode(
            decoder=self,
            output_time_major=True,
            impute_finished=False,
            maximum_iterations=maximum_iterations)
        return self.finalize(outputs, final_state)

    def compute_output(self, cell_output):
        """Computes the decoder outputs."""

        # Compute attention
        att_scores, attention_context = self.attention_fn(
            query=cell_output,
            keys=self.attention_keys,
            values=self.attention_values,
            values_length=self.attention_values_length)

        # TODO: Make this a parameter: We may or may not want this.
        # Transform attention context.
        # This makes the softmax smaller and allows us to synthesize information
        # between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        softmax_input = tf.contrib.layers.fully_connected(
            inputs=tf.concat([cell_output, attention_context], 1),
            num_outputs=self.cell.output_size,
            activation_fn=tf.nn.tanh,
            scope="attention_mix")

        # Softmax computation
        logits = tf.contrib.layers.fully_connected(
            inputs=softmax_input,
            num_outputs=self.vocab_size,
            activation_fn=None,
            scope="logits")

        return softmax_input, logits, att_scores, attention_context

    @property
    def batch_size(self):
        return tf.shape(nest.flatten([self.initial_state])[0])[0]

    @property
    def output_size(self):
        return AttentionDecoderOutput(
            logits=self.vocab_size,
            predicted_ids=tf.TensorShape([]),
            cell_output=self.cell.output_size,
            attention_scores=tf.shape(self.attention_values)[1:-1],
            attention_context=self.attention_values.get_shape()[-1])

    @property
    def output_dtype(self):
        return AttentionDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            cell_output=tf.float32,
            attention_scores=tf.float32,
            attention_context=tf.float32)

    def initialize(self, name=None):
        finished, first_inputs = self.helper.initialize()

        # Concat empty attention context
        attention_context = tf.zeros([
            tf.shape(first_inputs)[0],
            self.attention_values.get_shape().as_list()[-1]
        ])
        first_inputs = tf.concat([first_inputs, attention_context], 1)

        return finished, first_inputs, self.initial_state

    def step(self, time_, inputs, state, name=None):
        cell_output, cell_state = self.cell(inputs, state)
        cell_output_new, logits, attention_scores, attention_context = self.compute_output(cell_output)

        if self.reverse_scores_lengths is not None:
            attention_scores = tf.reverse_sequence(
                input=attention_scores,
                seq_lengths=self.reverse_scores_lengths,
                seq_dim=1,
                batch_dim=0
            )

        sample_ids = self.helper.sample(time=time_, outputs=logits, state=cell_state)

        outputs = AttentionDecoderOutput(
            logits=logits,
            predicted_ids=sample_ids,
            cell_output=cell_output_new,
            attention_scores=attention_scores,
            attention_context=attention_context)

        finished, next_inputs, next_state = self.helper.next_inputs(
            time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

        return outputs, next_state, next_inputs, finished


class PassThroughBridge(Configurable):
    """
    Passes the encoder state through to the decoder as-is. This bridge
    can only be used if encoder and decoder have the exact same state size, i.e.
    use the same RNN cell.

    All logic is contained in the `_create` method, which returns an
    initial state for the decoder.

    Args:
      encoder_outputs: A namedtuple that corresponds to the the encoder outputs.
      decoder_state_size: An integer or tuple of integers defining the
        state size of the decoder.
    """

    def __init__(self, encoder_outputs, decoder_state_size, params, mode):
        Configurable.__init__(self, params, mode)
        self.encoder_outputs = encoder_outputs
        self.decoder_state_size = decoder_state_size
        self.batch_size = tf.shape(
            nest.flatten(self.encoder_outputs.final_state)[0])[0]

    def __call__(self):
        """Runs the bridge function.

        Returns:
          An initial decoder_state tensor or tuple of tensors.
        """
        return self._create()

    def _create(self):
        nest.assert_same_structure(self.encoder_outputs.final_state,
                                   self.decoder_state_size)
        return self.encoder_outputs.final_state

    @staticmethod
    def default_params():
        return {}


class GreedyEmbeddingHelper(helper.Helper):
    """A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, max_time):
        """Initializer.

        Args:
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self.max_time = max_time

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return finished, self._start_inputs

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        # time_limit_exceed = math_ops.greater(time, self._max_time)
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)

        if self.max_time is not None:
            all_finished = finished = tf.equal(time, self.max_time)
        # is_finished = math_ops.logical_or(all_finished, time_limit_exceed)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return finished, next_inputs, state


def _total_tensor_depth(tensor):
    """Returns the size of a tensor without the first (batch) dimension"""
    return np.prod(tensor.get_shape().as_list()[1:])


class InitialStateBridge(Configurable):
    """A bridge that creates an initial decoder state based on the output
    of the encoder. This state is created by passing the encoder outputs
    through an additional layer to match them to the decoder state size.
    The input function remains unmodified.

    Args:
      encoder_outputs: A namedtuple that corresponds to the the encoder outputs.
      decoder_state_size: An integer or tuple of integers defining the
        state size of the decoder.
      bridge_input: Which attribute of the `encoder_outputs` to use for the
        initial state calculation. For example, "final_state" means that
        `encoder_outputs.final_state` will be used.
      activation_fn: An optional activation function for the extra
        layer inserted between encoder and decoder. A string for a function
        name contained in `tf.nn`, e.g. "tanh".
    """

    def __init__(self, encoder_outputs, decoder_state_size, params, mode):
        Configurable.__init__(self, params, mode)
        self.encoder_outputs = encoder_outputs
        self.decoder_state_size = decoder_state_size
        self.batch_size = tf.shape(
            nest.flatten(self.encoder_outputs.final_state)[0])[0]

        if not hasattr(encoder_outputs, self.params["bridge_input"]):
            raise ValueError("Invalid bridge_input not in encoder outputs.")

        self._bridge_input = getattr(encoder_outputs, self.params["bridge_input"])
        self._activation_fn = tf.identity

    def __call__(self):
        """Runs the bridge function.

        Returns:
          An initial decoder_state tensor or tuple of tensors.
        """
        return self._create()

    @staticmethod
    def default_params():
        return {
            "bridge_input": "final_state",
        }

    def _create(self):
        # Concat bridge inputs on the depth dimensions
        bridge_input = nest.map_structure(
            lambda x: tf.reshape(x, [self.batch_size, _total_tensor_depth(x)]),
            self._bridge_input)
        bridge_input_flat = nest.flatten([bridge_input])
        bridge_input_concat = tf.concat(bridge_input_flat, 1)

        state_size_splits = nest.flatten(self.decoder_state_size)
        total_decoder_state_size = sum(state_size_splits)

        # Pass bridge inputs through a fully connected layer layer
        initial_state_flat = tf.contrib.layers.fully_connected(
            inputs=bridge_input_concat,
            num_outputs=total_decoder_state_size,
            activation_fn=self._activation_fn)

        # Shape back into required state size
        initial_state = tf.split(initial_state_flat, state_size_splits, axis=1)
        return nest.pack_sequence_as(self.decoder_state_size, initial_state)


def get_output_projection_fn(size):
    def output_projection_fn(ids):
        return tf.one_hot(ids, size)

    return output_projection_fn


def build_model(batch_size, input_num_tokens, output_num_tokens, is_in_train_mode):
    mode_key = ModeKeys.TRAIN if is_in_train_mode else ModeKeys.EVAL
    inputs = tf.placeholder(tf.float32, [None, batch_size, input_num_tokens])
    input_length = tf.placeholder(tf.int32, [batch_size])
    # targets = tf.placeholder(tf.float32, [None, batch_size, output_num_tokens])
    target_labels = tf.placeholder(tf.int32, [None, batch_size])
    target_length = tf.placeholder(tf.int32, [batch_size])

    with variable_scope.variable_scope("Seq2Seq", reuse=tf.AUTO_REUSE):
        encoder_params = Encoder.default_params()
        encoder_params['rnn_cell']['num_layers'] = 2
        encoder = Encoder(encoder_params, mode_key, 'eeencoder')
        encoder_output = encoder(inputs, input_length, time_major=True)

        attention_params = AttentionLayerDot.default_params()
        attention = AttentionLayerDot(attention_params, mode_key)

        decoder_params = Decoder.default_params()
        decoder_params['rnn_cell']['num_layers'] = 1

        bridge_params = InitialStateBridge.default_params()
        decoder_state_size = decoder_params['rnn_cell']['cell_params']['num_units']
        decoder_state_size *= decoder_params['rnn_cell']['num_layers']
        bridge = InitialStateBridge(encoder_output, decoder_state_size, bridge_params, mode_key)

        attention_values = encoder_output.attention_values
        attention_values_length = encoder_output.attention_values_length
        attention_keys = encoder_output.outputs
        decoder = Decoder(
            params=decoder_params,
            mode=mode_key,
            vocab_size=output_num_tokens,
            attention_keys=attention_keys,
            attention_values=attention_values,
            attention_values_length=attention_values_length,
            attention_fn=attention,
            name='dddecoder',
        )
        # if is_in_train_mode:
        #     _helper = helper.TrainingHelper(
        #         inputs=targets,
        #         sequence_length=target_length,
        #         time_major=True,
        #     )
        # else:
        _helper = GreedyEmbeddingHelper(
            embedding=get_output_projection_fn(output_num_tokens),
            start_tokens=tf.fill([batch_size], output_num_tokens - 2),
            end_token=output_num_tokens - 1,
            max_time=tf.reduce_max(target_length),
        )
        targets = None
        decoder_initial_state = bridge()
        decoder_output, decoder_final_state = decoder(decoder_initial_state, _helper)

        # losses = seq2seq_losses.cross_entropy_sequence_loss(
        #     logits=decoder_output.logits,
        #     targets=target_labels,
        #     sequence_length=target_length,
        # )

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=decoder_output.logits,
            labels=target_labels,
        )

        outputs = tf.nn.softmax(decoder_output.logits)
        outputs = tf.argmax(outputs, axis=-1)

        # Calculate the average log perplexity
        loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(target_length))

    return inputs, input_length, targets, target_labels, target_length, outputs, loss


if __name__ == '__main__':
    build_model(20, 40, 30)
