import tensorflow as tf
from tensorflow import variable_scope
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from current_net_conf import *


class WordsEncoder:
    def __init__(self, last_state, all_states):
        self.last_state = last_state
        self.all_states = all_states


class WordsEncoderPlaceholders:
    def __init__(self, batch_size):
        with variable_scope('placeholders'):
            self.words_rules_seq = tf.placeholder(tf.int32, [None, batch_size], 'rules_sequence')
            self.words_rules_seq_len = tf.placeholder(tf.int32, [batch_size], 'rules_sequence_length')


def build_words_encoder(rules_count, batch_size=BATCH_SIZE):
    with variable_scope('words_encoder') as scope:
        placeholders = WordsEncoderPlaceholders(batch_size)
        embedding = tf.get_variable(
            name='rules_embedding',
            shape=[rules_count, RULES_ENCODER_INPUT_SIZE],
            dtype=tf.float32
        )

        prepared_inputs = tf.nn.embedding_lookup(embedding, placeholders.words_rules_seq)

        encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(ss) for ss in RULES_ENCODER_STATE_SIZE_BY_LAYER]
        encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(ss) for ss in RULES_ENCODER_STATE_SIZE_BY_LAYER]

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
            sequence_length=placeholders.words_rules_seq_len,
            time_major=True,
            dtype=tf.float32,
            scope=scope,
        )

        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

        encoder = WordsEncoder(encoder_result, encoder_output)

        return encoder, placeholders
