import tensorflow as tf


def print_shape(tensor, name):
    return tf.Print(tensor, [tf.shape(tensor)], name + ' ')


def print_value(tensor, name):
    return tf.Print(tensor, [tensor], name + ' ', summarize=100)


def print_if_nan(tensor, name):
    is_nan = tf.reduce_any(tf.is_nan(tensor))
    return tf.cond(is_nan, lambda: tf.Print(tensor, [tensor], 'NAN ' + name + ' '), lambda: tensor)


def fix_r_index_keys(r_index):
    return {int(key): value for key, value in r_index.items()}


def tf_accuracy(predicted, target, mask=None, need_round=False, shape=None):
    if mask is not None:
        mask = tf.cast(mask, tf.bool)
        predictions = tf.boolean_mask(predicted, mask)
        targets = tf.boolean_mask(target, mask)
    else:
        if shape is None:
            shape = [-1]
        predictions = tf.reshape(predicted, shape)
        targets = tf.reshape(target, shape)
    empty_tensor = tf.equal(tf.size(targets), 0)
    if need_round:
        predictions = tf.round(predictions)
        targets = tf.round(targets)
    predictions = tf.to_int32(predictions)
    targets = tf.to_int32(targets)
    eq_elements = tf.equal(predictions, targets)
    eq_elements = tf.to_int32(eq_elements)
    eq_count = tf.reduce_sum(eq_elements, axis=0)
    tensor_length = tf.shape(targets)[0]
    eq_count = tf.to_float(eq_count)
    length = tf.to_float(tensor_length)
    accuracy = eq_count / length

    real_accuracy = tf.cond(empty_tensor, lambda: 1.0, lambda: accuracy)

    return real_accuracy
