import tensorflow as tf
from tensorflow import variable_scope
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from current_net_conf import *
from model.tf_utils import tf_conditional_lookup


class WordsEncoder:
    def __init__(self, last_state, all_states):
        self.last_state = last_state
        self.all_states = all_states

    def fetch_all(self):
        return [
            self.last_state,
            self.all_states
        ]


class WordsEncoderPlaceholders:
    def __init__(self, batch_size):
        with variable_scope('placeholders'):
            self.parent_rules_seq_with_pc = tf.placeholder(tf.int32, [None, batch_size], 'rules_sequence')
            self.nodes_seq_with_pc = tf.placeholder(tf.int32, [None, batch_size], 'rules_sequence')
            self.rules_seq_with_pc = tf.placeholder(tf.int32, [None, batch_size], 'rules_sequence')
            self.rules_seq_with_pc_len = tf.placeholder(tf.int32, [batch_size], 'rules_sequence_length')

    def feed(self, rules, rules_len, parent_rules, nodes):
        return {
            self.rules_seq_with_pc: rules,
            self.rules_seq_with_pc_len: rules_len,
            self.parent_rules_seq_with_pc: parent_rules,
            self.nodes_seq_with_pc: nodes
        }


def build_words_encoder(rules_count, nodes_count, batch_size=BATCH_SIZE):
    with variable_scope('words_encoder') as scope:
        placeholders = WordsEncoderPlaceholders(batch_size)

        rules_embedding = tf.get_variable(
            name='rules_embedding',
            shape=[rules_count, RULES_ENCODER_INPUT_SIZE],
            dtype=tf.float32
        )
        nodes_embedding = tf.get_variable(
            name='nodes_embedding',
            shape=[nodes_count, RULES_ENCODER_INPUT_SIZE],
            dtype=tf.float32
        )

        rules = tf.nn.embedding_lookup(rules_embedding, placeholders.rules_seq_with_pc)

        parent_rules_cond = tf.not_equal(placeholders.parent_rules_seq_with_pc, -1)
        parent_rules = tf_conditional_lookup(parent_rules_cond, rules_embedding, placeholders.parent_rules_seq_with_pc)

        nodes_cond = tf.not_equal(placeholders.nodes_seq_with_pc, -1)
        nodes = tf_conditional_lookup(nodes_cond, nodes_embedding, placeholders.nodes_seq_with_pc)

        prepared_inputs = tf.concat([rules, parent_rules, nodes], axis=-1)

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
            sequence_length=placeholders.rules_seq_with_pc_len,
            time_major=True,
            dtype=tf.float32,
            scope=scope,
        )

        final_encoder_state_fw = encoder_state_fw[-1]
        final_encoder_state_bw = encoder_state_bw[-1]

        encoder_result = tf.concat([final_encoder_state_fw, final_encoder_state_bw], 1)

        encoder = WordsEncoder(encoder_result, encoder_output)

        return encoder, placeholders
