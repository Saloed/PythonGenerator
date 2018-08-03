import tensorflow as tf
from tensorflow import variable_scope
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from current_net_conf import *
from utils import dict_to_object


def build_words_encoder(rules_count, rules_decoder_placeholders):
    with variable_scope('words_encoder') as scope:
        rules_seq = rules_decoder_placeholders['rules_target']
        rules_seq_length = rules_decoder_placeholders['rules_sequence_length']

        embedding = tf.get_variable(
            name='rules_embedding',
            shape=[rules_count, ENCODER_INPUT_SIZE],
            dtype=tf.float32
        )

        prepared_inputs = tf.nn.embedding_lookup(embedding, rules_seq)

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
            sequence_length=rules_seq_length,
            time_major=True,
            dtype=tf.float32,
            scope=scope,
        )

        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

        placeholders = {}

        encoder = dict_to_object({
            'last_state': encoder_result,
            'all_states': encoder_output
        }, placeholders)

        return encoder, placeholders


def build_words_encoder_with_rules(rules_count):
    with variable_scope('words_encoder'):
        with variable_scope('placeholders'):
            rules_target = tf.placeholder(tf.int32, [None, 1], 'rules_sequence')
            rules_sequence_length = tf.placeholder(tf.int32, [1], 'rules_sequence_length')

            new_placeholders = {
                'rules_target': rules_target,
                'rules_sequence_length': rules_sequence_length
            }
    encoder, placeholders = build_words_encoder(rules_count, new_placeholders)
    new_placeholders.update(placeholders)
    return encoder, new_placeholders
