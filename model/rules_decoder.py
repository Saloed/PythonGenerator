import tensorflow as tf
from tensorflow import variable_scope

from current_net_conf import *
from model import utils
from model.rnn_with_dropout import MultiRnnWithDropout
from utils import dict_to_object


def build_rules_decoder(encoder_last_state, rules_count):
    with variable_scope("rules_decoder") as scope:
        with variable_scope('placeholders'):
            rules_target = tf.placeholder(tf.int32, [None, BATCH_SIZE], 'rules_target')
            rules_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE], 'rules_target_length')

        if scope.caching_device is None:
            scope.set_caching_device(lambda op: op.device)

        maximum_iterations = tf.reduce_max(rules_sequence_length)  # just for simplify training

        rnn_cell = MultiRnnWithDropout(RULES_DECODER_LAYERS, DECODER_STATE_SIZE)
        outputs_projection = tf.layers.Dense(rules_count, name='rules_output_projection')

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_state = rnn_cell.initial_state(encoder_last_state)
        initial_inputs = rnn_cell.zero_initial_inputs(rules_count)

        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=maximum_iterations,
            element_shape=[BATCH_SIZE, rules_count]
        )

        def condition(_time, unused_outputs_ta, unused_state, unused_inputs):
            return _time < maximum_iterations

        def body(time, outputs_ta, state, inputs):
            cell_output, cell_state = rnn_cell(inputs, state)
            projected_outputs = outputs_projection(cell_output)
            outputs_ta = outputs_ta.write(time, projected_outputs)
            return time + 1, outputs_ta, cell_state, projected_outputs

        _, final_outputs_ta, final_state, *_ = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs
            ]
        )

        # time, batch, rules_count
        outputs_logits = final_outputs_ta.stack()
        outputs = tf.nn.softmax(outputs_logits)

        placeholders = {
            'rules_sequence_length': rules_sequence_length,
            'rules_target': rules_target
        }

        decoder = dict_to_object({
            'rules': outputs,
            'rules_logits': outputs_logits,
            'rules_sequence_length': rules_sequence_length
        }, placeholders)

        return decoder, placeholders


def build_rules_loss(decoder):
    with variable_scope("rules_decoder_loss"):
        raw_rules_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder.rules_target,
            logits=decoder.rules_logits
        )
        loss_mask = tf.sequence_mask(decoder.rules_sequence_length, dtype=tf.float32)
        loss_mask = tf.transpose(loss_mask, [1, 0])
        masked_rules_loss = raw_rules_loss * loss_mask
        rules_loss = tf.reduce_sum(masked_rules_loss)

    with variable_scope('stats'):
        scaled_logits = tf.nn.softmax(decoder.rules_logits)
        results = tf.argmax(scaled_logits, axis=-1)
        rules_accuracy = utils.tf_accuracy(
            predicted=results,
            target=(decoder.rules_target),
            mask=loss_mask
        )

    stats = {
        'rules_accuracy': rules_accuracy
    }

    loss = {
        'rules_loss': rules_loss,
    }
    return loss, stats
