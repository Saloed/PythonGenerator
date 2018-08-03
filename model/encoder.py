import tensorflow as tf
from tensorflow import variable_scope
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from current_net_conf import *
from utils import dict_to_object


def build_encoder_batch_specific(query_tokens_count, batch_size):
    with variable_scope('query_encoder') as scope:
        with variable_scope('placeholders'):
            query_ids = tf.placeholder(tf.int32, [None, batch_size], 'query_ids')
            query_length = tf.placeholder(tf.int32, [batch_size], 'query_length')

        embedding = tf.get_variable(
            name='input_embedding',
            shape=[query_tokens_count, ENCODER_INPUT_SIZE],
            dtype=tf.float32
        )

        prepared_inputs = tf.nn.embedding_lookup(embedding, query_ids)
        encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(ENCODER_STATE_SIZE) for _ in range(ENCODER_LAYERS)]
        encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(ENCODER_STATE_SIZE) for _ in range(ENCODER_LAYERS)]

        encoder_fw_internal_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                cell=cell,
                output_keep_prob=ENCODER_DROPOUT_PROB,
                state_keep_prob=ENCODER_DROPOUT_PROB
            )
            for cell in encoder_fw_internal_cells
        ]
        encoder_bw_internal_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                cell=cell,
                output_keep_prob=ENCODER_DROPOUT_PROB,
                state_keep_prob=ENCODER_DROPOUT_PROB
            )
            for cell in encoder_bw_internal_cells
        ]

        encoder_output, encoder_state_fw, encoder_state_bw = stack_bidirectional_dynamic_rnn(
            cells_fw=encoder_fw_internal_cells,
            cells_bw=encoder_bw_internal_cells,
            inputs=prepared_inputs,
            sequence_length=query_length,
            time_major=True,
            dtype=tf.float32,
            scope=scope,
        )

        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

        placeholders = {
            'query_ids': query_ids,
            'query_length': query_length,
        }

        encoder = dict_to_object({
            'last_state': encoder_result,
            'all_states': encoder_output
        }, placeholders)

        return encoder, placeholders


def build_encoder(query_tokens_count):
    return build_encoder_batch_specific(query_tokens_count, BATCH_SIZE)

