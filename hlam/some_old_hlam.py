
from some_net_stuff import Encoder
from some_net_stuff.BatchBuilder import generate_batches
from some_net_stuff.NetBuilder import build_net
from some_net_stuff.TFParameters import init_params

def pretrain_with_negative_samples(tokens, code_inputs, characters, indexed_text_inputs):
    params, emb_indexes = init_params(tokens)
    net = build_net(params)
    negative_net = build_net(params)
    code_batches = generate_batches(code_inputs, emb_indexes, net, 0.8)
    negative_code_batches = generate_batches(code_inputs, emb_indexes, negative_net, 1.0)

    text_input_pc, encoder_out, empty_token_idx = Encoder.get_net(len(characters))
    text_input_batches = Encoder.generate_batches(indexed_text_inputs, text_input_pc, empty_token_idx)

    distance_delta = tf.constant(5.0, tf.float32)

    # negative_net_out = tf.stop_gradient(negative_net.out)
    negative_net_out = negative_net.out

    positive_net_out = net.out
    positive_distance = tf.norm(encoder_out - positive_net_out, axis=1)

    # negative_distance = tf.stop_gradient(tf.norm(encoder_out - negative_net_out))
    negative_distance = tf.norm(encoder_out - negative_net_out, axis=1)

    # tf.summary.histogram('encoder_out', encoder_out)
    # tf.summary.histogram('positive_out', positive_net_out)
    # tf.summary.histogram('negative_out', negative_net_out)

    # loss = positive_distance / (negative_distance + 1e-6)

    loss = positive_distance + distance_delta - negative_distance
    loss = tf.nn.relu(loss)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('positive', tf.reduce_mean(positive_distance))
    tf.summary.scalar('negative', tf.reduce_mean(negative_distance))
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()
    weights = [
        var
        for var in tf.trainable_variables()
        if len(var.shape) > 1 and var.shape[0] > 1
           and not var.name.startswith('rnn')
    ]
    regularized_weights = [
        tf.nn.l2_loss(w)
        for w in weights
    ]
    l2_loss = tf.reduce_sum(regularized_weights)
    # updates = tf.train.AdamOptimizer().minimize(loss + 1e-3 * l2_loss)
    updates = tf.train.RMSPropOptimizer(
        learning_rate=0.001,
        decay=0.1,
        momentum=0.9,
        centered=True,
    ).minimize(loss + 1e-3 * l2_loss)

    validation_code, train_code = split_batches(code_batches, 10)
    valid_negative, train_negative = split_batches(negative_code_batches, 10)
    validation_text, train_text = split_batches(text_input_batches, 10)

    initializer = tf.global_variables_initializer()
    for retry_num in range(1):
        # plot_axes, plot = new_figure(retry_num, 100, 2)
        saver = tf.train.Saver(max_to_keep=10)
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter('models', sess.graph)
            sess.run(initializer)
            try:
                for train_epoch in range(100):
                    print(f'start epoch {train_epoch}')
                    res = []
                    for j, feed in enumerate(make_samples_feed(train_text, train_code, train_negative, 3)):
                        err, summary, _ = sess.run(fetches=[loss, summaries, updates], feed_dict=feed)
                        summary_writer.add_summary(summary, train_epoch * 3000 + j)
                        res.append(float(err))
                        # if (j + 1) % 50 == 0:
                        #     print(f'iteration {j}')

                    tr_loss = np.mean(res)
                    print('valid epoch')
                    res = []
                    for j, feed in enumerate(make_samples_feed(validation_text, validation_code, valid_negative, 1)):
                        err, *_ = sess.run(fetches=[loss], feed_dict=feed)
                        res.append(float(err))

                    v_loss = np.mean(res)

                    print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')
                    saver.save(sess, 'models/model', train_epoch)

                    # update_figure(plot, plot_axes, train_epoch, v_loss, tr_loss)
            except Exception as ex:
                print(ex)
                # print_tb(ex.__traceback__)


def seq2seq_input_check(num_text_inputs, indexed_text_inputs):
    batch_size = 10
    num_text_inputs_with_end = num_text_inputs + 1
    inputs, input_length, targets, target_labels, target_length, loss = Seq2Seq.build_model(
        batch_size,
        num_text_inputs_with_end,
        num_text_inputs_with_end,
    )
    text_batches = group_text_by_batches(indexed_text_inputs, batch_size, num_text_inputs_with_end, True)
    valid, train = split_batches(text_batches, 10)
    updates = tf.train.AdamOptimizer().minimize(loss)
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    initializer = tf.global_variables_initializer()
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()

    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        summary_writer = tf.summary.FileWriter('models', sess.graph)
        sess.run(initializer)
        try:
            for train_epoch in range(100):
                print(f'start epoch {train_epoch}')
                res = []
                batch_count = len(train)
                for j, (sample, sample_lens) in enumerate(train):
                    sample, sample_embs = get_sample_embs(sample, num_text_inputs_with_end)
                    err, summary, _ = sess.run(fetches=[loss, summaries, updates], feed_dict={
                        inputs: sample_embs,
                        input_length: sample_lens,
                        targets: sample_embs,
                        target_labels: sample,
                        target_length: sample_lens,
                    })
                    summary_writer.add_summary(summary, train_epoch * batch_count + j)

                    res.append(float(err))
                    batch_number = j + 1
                    if batch_number % 100 == 0:
                        percent = int(j / batch_count * 100)
                        print(f'Complete {percent}')

                tr_loss = np.mean(res)
                print('valid epoch')
                res = []
                for sample, sample_lens in valid:
                    sample, sample_embs = get_sample_embs(sample, num_text_inputs_with_end)
                    err, *_ = sess.run(fetches=[loss], feed_dict={
                        inputs: sample_embs,
                        input_length: sample_lens,
                        targets: sample_embs,
                        target_labels: sample,
                        target_length: sample_lens,
                    })
                    res.append(float(err))

                v_loss = np.mean(res)

                print(f'epoch {train_epoch} train {tr_loss} valid {v_loss}')

        except Exception as ex:
            print(ex)
            print_tb(ex.__traceback__)
