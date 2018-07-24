import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model import utils
from utils import dict_to_object


def build_copy_mechanism(encoder_states, decoder_states, generate_tokens_count):
    with variable_scope('copy_mechanism'):
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
                shape=[DECODER_STATE_SIZE],
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
        copy_scores = tf.nn.softmax(copy_scores_logits, name='copy_scores_softmax')

        # decoder_time, batch
        max_copy_score = tf.reduce_max(copy_scores, axis=2, name='max_copy_score')

        with variable_scope('decoder_state_translation'):
            decoder_state_translation = tf.get_variable(
                name='kernel',
                shape=[DECODER_STATE_SIZE]
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

        return dict_to_object({
            'copy_scores_logits': copy_scores_logits,
            'copy_scores': copy_scores,

            'generate_or_copy_score_logits': generate_or_copy_score_logits,
            'generate_or_copy_score': generate_or_copy_score,

            'generated_tokens_logits': generated_tokens_logits,
            'generated_tokens': generated_tokens,
        })


def build_words_loss(copy_mechanism, decoder_placeholders):
    with variable_scope('copy_mechanism_loss'):
        with variable_scope('placeholders'):
            generate_token_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'token_target')
            generate_token_target_mask = tf.placeholder(tf.float32, [None, BATCH_SIZE], 'token_target_mask')

            copy_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'copy_target')
            copy_target_mask = tf.placeholder(tf.float32, [None, BATCH_SIZE], 'copy_target_mask')

            generate_or_copy_target = tf.placeholder(tf.float32, [None, BATCH_SIZE, 2], 'generate_or_copy_target')

        loss_mask = tf.sequence_mask(decoder_placeholders['words_sequence_length'], dtype=tf.float32)
        loss_mask = tf.transpose(loss_mask, [1, 0])

        raw_generate_or_copy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=generate_or_copy_target,
            logits=copy_mechanism.generate_or_copy_score_logits
        )
        raw_generate_or_copy_loss = tf.reduce_sum(raw_generate_or_copy_loss, axis=2)
        masked_generate_or_copy_loss = raw_generate_or_copy_loss * loss_mask

        generate_or_copy_loss = tf.reduce_sum(masked_generate_or_copy_loss)

        raw_generate_token_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=generate_token_target,
            logits=copy_mechanism.generated_tokens_logits
        )

        masked_generate_token_loss = raw_generate_token_loss * generate_token_target_mask

        generate_token_loss = tf.reduce_sum(masked_generate_token_loss)

        raw_copy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=copy_target,
            logits=copy_mechanism.copy_scores_logits
        )
        masked_copy_loss = raw_copy_loss * copy_target_mask

        copy_loss = tf.reduce_sum(masked_copy_loss)

    with variable_scope('stats'):
        generate_logits = copy_mechanism.generated_tokens_logits
        generate_logits_scaled = tf.nn.softmax(generate_logits)
        generate_tokens = tf.argmax(generate_logits_scaled, axis=-1)
        generate_tokens_accuracy = utils.tf_accuracy(
            predicted=generate_tokens,
            target=generate_token_target,
            mask=generate_token_target_mask
        )

        copy_logits = copy_mechanism.copy_scores_logits
        copy_logits_scaled = tf.nn.softmax(copy_logits)
        copy_indices = tf.argmax(copy_logits_scaled, axis=-1)
        copy_accuracy = utils.tf_accuracy(
            predicted=copy_indices,
            target=copy_target,
            mask=copy_target_mask
        )

        decision_logits = copy_mechanism.generate_or_copy_score_logits
        decision_logits_scaled = tf.nn.sigmoid(decision_logits)
        decision_accuracy = utils.tf_accuracy(
            predicted=decision_logits_scaled,
            target=generate_or_copy_target,
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

    placeholders = {
        'generate_token_target': generate_token_target,
        'generate_token_target_mask': generate_token_target_mask,

        'copy_target': copy_target,
        'copy_target_mask': copy_target_mask,

        'generate_or_copy_target': generate_or_copy_target
    }

    return losses, stats, placeholders
