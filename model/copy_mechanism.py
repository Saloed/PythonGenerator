import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model.tf_utils import *
from model.words_decoder import WordsDecoderPlaceholders


class CopyMechanism:
    def __init__(self, copy_scores, generate_or_copy_score, generated_tokens,
                 copy_scores_logits=None, generate_or_copy_score_logits=None, generated_tokens_logits=None):
        self.copy_scores = copy_scores
        self.generate_or_copy_score = generate_or_copy_score
        self.generated_tokens = generated_tokens

        self.copy_scores_logits = copy_scores_logits
        self.generate_or_copy_score_logits = generate_or_copy_score_logits
        self.generated_tokens_logits = generated_tokens_logits

    def copy_mechanism_fetch(self):
        return [
            self.copy_scores,
            self.generate_or_copy_score,
            self.generated_tokens,

            self.copy_scores_logits,
            self.generate_or_copy_score_logits,
            self.generated_tokens_logits,
        ]


class CopyMechanismPlaceholders:
    def __init__(self):
        with variable_scope('placeholders'):
            self.generate_token_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'token_target')
            self.generate_token_target_mask = tf.placeholder(tf.bool, [None, BATCH_SIZE], 'token_target_mask')

            self.copy_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'copy_target')
            self.copy_target_mask = tf.placeholder(tf.bool, [None, BATCH_SIZE], 'copy_target_mask')

            self.generate_or_copy_target = tf.placeholder(tf.float32, [None, BATCH_SIZE, 2], 'generate_or_copy_target')


class CopyMechanismPlaceholdersSingleStep:
    def __init__(self):
        with variable_scope('placeholders'):
            self.query_encoder_states = tf.placeholder(tf.float32, [None, 1, WORDS_DECODER_STATE_SIZE])

    def copy_mechanism_feed(self, query_states):
        return {
            self.query_encoder_states: query_states
        }


def build_copy_mechanism(encoder_states, decoder_states, generate_tokens_count):
    with variable_scope('copy_mechanism'):
        placeholders = CopyMechanismPlaceholders()

        # batch, time, encoder state size
        _encoder_states = tf.transpose(encoder_states, [1, 0, 2], name='transpose_encoder_states_batch_major')
        # 1, batch, encoder time, encoder state size
        prepared_encoder_states = tf.expand_dims(_encoder_states, 0, name='expand_encoder_states_at_0')

        # decoder time, batch, 1, decoder state size
        prepared_decoder_states = tf.expand_dims(decoder_states, 2, name='expand_decoder_states_at_2')

        assert prepared_encoder_states.shape[-1] == prepared_decoder_states.shape[-1]

        # decoder time, batch, encoder time, state size
        combined_states = tf.add(prepared_encoder_states, prepared_decoder_states, name='combine_states')
        bounded_combined_states = tf.nn.tanh(combined_states, name='bound_states')

        with variable_scope('copy'):
            copy_weight = tf.get_variable(
                name='kernel',
                shape=[WORDS_DECODER_STATE_SIZE],
                dtype=tf.float32
            )

            copy_bias = tf.get_variable(
                name='bias',
                shape=[],
                dtype=tf.float32
            )

        # decoder_time, batch, encoder_time
        copy_scores_logits = tf.einsum('abci,i->abc', bounded_combined_states, copy_weight, name='copy_scores_logits')
        copy_scores_logits = tf.add(copy_scores_logits, copy_bias, name='add_bias')

        with variable_scope('copy_scores_softmax'):
            copy_scores = tf.nn.softmax(copy_scores_logits)

        # decoder_time, batch
        max_copy_score = tf.reduce_max(copy_scores, axis=2, name='max_copy_score')

        with variable_scope('decoder_state_translation'):
            decoder_state_translation = tf.get_variable(
                name='kernel',
                shape=[WORDS_DECODER_STATE_SIZE]
            )

        # decoder_time, batch
        generate_probability = tf.einsum('ijk,k->ij', decoder_states, decoder_state_translation,
                                         name='translate_decoder_state')

        # decoder_time, batch, 2
        generate_and_copy_probs = tf.stack([max_copy_score, generate_probability], axis=2,
                                           name='stack_decoder_state_and_max_copy_score')

        generate_or_copy_score_logits = tf.layers.dense(generate_and_copy_probs, 2, name='generate_or_copy')
        generate_or_copy_score = tf.nn.sigmoid(generate_or_copy_score_logits, name='generate_or_copy_sigmoid')

        generated_tokens_logits = tf.layers.dense(decoder_states, generate_tokens_count, name='tokens_projection')
        generated_tokens = tf.nn.softmax(generated_tokens_logits, name='generated_tokens_softmax')

        cm = CopyMechanism(copy_scores, generate_or_copy_score, generated_tokens,
                           copy_scores_logits, generate_or_copy_score_logits, generated_tokens_logits)
        return cm, placeholders


def build_copy_mechanism_single_step(decoder_states, generate_tokens_count):
    with variable_scope('copy_mechanism'):
        placeholders = CopyMechanismPlaceholdersSingleStep()

    expanded_decoder_states = tf.expand_dims(decoder_states, 0)
    copy_mechanism, _ = build_copy_mechanism(placeholders.query_encoder_states, expanded_decoder_states,
                                             generate_tokens_count)
    return copy_mechanism, placeholders


def build_words_loss(copy_mechanism, decoder_placeholders, pc):
    # type: (CopyMechanism, WordsDecoderPlaceholders, CopyMechanismPlaceholders) -> [dict, dict]

    with variable_scope('copy_mechanism_loss'):
        loss_mask = tf_length_mask(decoder_placeholders.words_sequence_length)
        raw_generate_or_copy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=pc.generate_or_copy_target,
            logits=copy_mechanism.generate_or_copy_score_logits
        )
        raw_generate_or_copy_loss = tf.reduce_sum(raw_generate_or_copy_loss, axis=2)
        generate_or_copy_loss = tf_mask_gracefully(raw_generate_or_copy_loss, loss_mask, sum_result=True)

        raw_generate_token_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=pc.generate_token_target,
            logits=copy_mechanism.generated_tokens_logits
        )
        generate_token_loss = tf_mask_gracefully(raw_generate_token_loss, pc.generate_token_target_mask,
                                                 sum_result=True)

        raw_copy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=pc.copy_target,
            logits=copy_mechanism.copy_scores_logits
        )
        copy_loss = tf_mask_gracefully(raw_copy_loss, pc.copy_target_mask, sum_result=True)

    with variable_scope('stats'):
        generate_logits = copy_mechanism.generated_tokens_logits
        generate_logits_scaled = tf.nn.softmax(generate_logits)
        generate_tokens = tf.argmax(generate_logits_scaled, axis=-1)
        generate_tokens_accuracy = tf_accuracy(
            predicted=generate_tokens,
            target=pc.generate_token_target,
            mask=pc.generate_token_target_mask
        )

        copy_logits = copy_mechanism.copy_scores_logits
        copy_logits_scaled = tf.nn.softmax(copy_logits)
        copy_indices = tf.argmax(copy_logits_scaled, axis=-1)
        copy_accuracy = tf_accuracy(
            predicted=copy_indices,
            target=pc.copy_target,
            mask=pc.copy_target_mask
        )

        decision_logits = copy_mechanism.generate_or_copy_score_logits
        decision_logits_scaled = tf.nn.sigmoid(decision_logits)
        decision_accuracy = tf_accuracy(
            predicted=decision_logits_scaled,
            target=pc.generate_or_copy_target,
            mask=loss_mask,
            need_round=True,
            shape=[-1, 2]
        )

    stats = {
        'generate_tokens_accuracy': generate_tokens_accuracy,
        'copy_accuracy': copy_accuracy,
        'decision_accuracy': decision_accuracy
    }

    losses = {
        'generate_or_copy_loss': generate_or_copy_loss,
        'generate_token_loss': generate_token_loss,
        'copy_loss': copy_loss
    }

    return losses, stats
