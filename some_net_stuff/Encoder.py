import tensorflow as tf
from tensorflow.contrib import rnn as tf_rnn

from some_net_stuff.TFParameters import BATCH_SIZE

NUM_LAYERS = 1
NUM_FEATURES = 100


def build_model(sequence, num_tokens):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', [num_tokens, NUM_FEATURES])

    def get_lstm_cell():
        return tf_rnn.GRUCell(
            num_units=NUM_FEATURES,
        )

    sequence = tf.transpose(sequence, [1, 0])

    cells_fwd = [get_lstm_cell() for _ in range(NUM_LAYERS)]
    cells_bwd = [get_lstm_cell() for _ in range(NUM_LAYERS)]
    state_fwd = [cell.zero_state(BATCH_SIZE, tf.float32) for cell in cells_fwd]
    state_bwd = [cell.zero_state(BATCH_SIZE, tf.float32) for cell in cells_bwd]

    rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

    with tf.variable_scope('rnn') as vs:
        outputs, _, _ = tf_rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fwd,
            cells_bw=cells_bwd,
            inputs=rnn_inputs,
            initial_states_fw=state_fwd,
            initial_states_bw=state_bwd,
            scope=vs,
            time_major=True,
        )
        last_output = outputs[-1]
        transformed = tf.contrib.layers.fully_connected(last_output, 30, scope=vs, activation_fn=tf.nn.tanh,)
    return transformed, outputs


def get_net(num_tokens):
    inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='text_input')
    outputs = build_model(inputs, num_tokens + 1)
    return inputs, outputs, num_tokens


def allign_batch(batch, empty_token_idx):
    lens = [len(b) for b in batch]
    max_len = max(lens)
    return [
        sample + [empty_token_idx for _ in range(max_len - sample_len)]
        for sample, sample_len in zip(batch, lens)
    ]


def generate_batches(data_set, input_pc, empty_token_idx):
    size = len(data_set) // BATCH_SIZE
    batches = []
    for j in range(size):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        d = allign_batch(d, empty_token_idx)
        feed = {input_pc: d}
        # for i in range(BATCH_SIZE):
        #     feed.update(pc[i].assign(prepare_batch(d[i], emb_indexes)))
        batches.append(feed)
    return batches


def test_encoder():
    sequence = tf.placeholder(tf.int32, [BATCH_SIZE, None])
    encoder_out, encoder_attention_states = build_model(sequence, 15)

if __name__ == '__main__':
    test_encoder()
