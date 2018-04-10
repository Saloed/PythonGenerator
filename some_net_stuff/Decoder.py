import tensorflow as tf

from some_net_stuff.Rnn import GRUCellWithAttention, dynamic_rnn_with_states
from some_net_stuff import Encoder
from some_net_stuff.TFParameters import BATCH_SIZE


def get_model(attention, encoder_out, num_tokens, sequence_end_marker, non_terminal_mask):
    end_marker = tf.constant(sequence_end_marker)

    zero_array = tf.zeros([num_tokens])

    def node_is_not_terminal(node):
        return (node * non_terminal_mask) == zero_array

    cell = GRUCellWithAttention(num_tokens, attention)

    def process_tree(initial_state, initial_input):
        output, output_ta, state, state_ta = dynamic_rnn_with_states(
            cell=cell,
            max_time_steps=10,
            sequence_end_marker=end_marker,
            initial_state=initial_state,
            initial_input=initial_input,
        )

        i = tf.constant(0, dtype=tf.int32, name="i")
        size = output_ta.size()
        last_out = tf.zeros([num_tokens])

        def resolve_subtree(_i, _output_ta, _state_ta, _last_out):
            _last_out = _output_ta.read(_i)
            _state = _state_ta.read(_i)

            def _resolve_subtree():
                return process_tree(_state, _last_out)

            tf.control_flow_ops.cond(
                pred=node_is_not_terminal(_last_out),
                true_fn=_resolve_subtree,
                false_fn=None
            )

        tf.control_flow_ops.while_loop(
            cond=tf.logical_and(i < size, last_out != end_marker),
            loop_vars=(i, output_ta, state_ta, last_out),
            body=resolve_subtree,
        )

        return output_ta


def tf_array_test():
    sequence = tf.placeholder(tf.int32, [BATCH_SIZE, None])
    encoder_out, encoder_attention_states = Encoder.build_model(sequence, 15)
    get_model(encoder_attention_states, encoder_out)
    # cell = GRUCellWithAttention(30, encoder_attention_states)
    # decoder_inputs = tf.ones([15, BATCH_SIZE, 40])
    # outputs, state = tf.nn.dynamic_rnn(
    #     cell,
    #     inputs=decoder_inputs,
    #     initial_state=encoder_out,
    #     dtype=tf.float32,
    #     time_major=True
    # )
    # a = outputs


if __name__ == '__main__':
    tf_array_test()
