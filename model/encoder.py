import tensorflow as tf
from tensorflow import variable_scope
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from current_net_conf import *


class QueryEncoder:
    def __init__(self, last_state, all_states):
        self.last_state = last_state
        self.all_states = all_states


class QueryEncoderPlaceholders:
    def __init__(self, batch_size):
        with variable_scope('placeholders'):
            self.query_ids = tf.placeholder(tf.int32, [None, batch_size], 'query_ids')
            self.query_length = tf.placeholder(tf.int32, [batch_size], 'query_length')


def build_query_encoder_for_rules(query_tokens_count, batch_size):
    return _build_query_encoder(
        query_tokens_count=query_tokens_count,
        name_prefix='',
        state_size=RULES_QUERY_ENCODER_STATE_SIZE,
        emb_size=RULES_QUERY_ENCODER_INPUT_SIZE,
        num_layers=RULES_QUERY_ENCODER_LAYERS,
        batch_size=batch_size
    )


def build_query_encoder_for_words(query_tokens_count, batch_size):
    return _build_query_encoder(
        query_tokens_count=query_tokens_count,
        name_prefix='words_',
        state_size=WORDS_QUERY_ENCODER_STATE_SIZE_BY_LAYER,
        emb_size=WORDS_QUERY_ENCODER_INPUT_SIZE,
        num_layers=WORDS_QUERY_ENCODER_LAYERS,
        batch_size=batch_size
    )


def _build_query_encoder(query_tokens_count, name_prefix, state_size, emb_size, num_layers, batch_size):
    with variable_scope(name_prefix + 'query_encoder') as scope:

        placeholders = QueryEncoderPlaceholders(batch_size)
        query_ids, query_length = placeholders.query_ids, placeholders.query_length

        embedding = tf.get_variable(
            name='input_embedding',
            shape=[query_tokens_count, emb_size],
            dtype=tf.float32
        )

        prepared_inputs = tf.nn.embedding_lookup(embedding, query_ids)
        if isinstance(state_size, list):
            encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(ss) for ss in state_size]
            encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(ss) for ss in state_size]
        else:
            encoder_fw_internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(num_layers)]
            encoder_bw_internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(num_layers)]

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
        encoder = QueryEncoder(encoder_result, encoder_output)

        return encoder, placeholders
