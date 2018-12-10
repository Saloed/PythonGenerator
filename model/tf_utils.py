import tensorflow as tf
from tensorflow import variable_scope


def tf_print_shape(tensor, name):
    return tf.Print(tensor, [tf.shape(tensor)], name + ' ')


def tf_print_value(tensor, name):
    return tf.Print(tensor, [tensor], name + ' ', summarize=100)


def tf_print_if_nan(tensor, name):
    is_nan = tf.reduce_any(tf.is_nan(tensor))
    return tf.cond(is_nan, lambda: tf.Print(tensor, [tensor], 'NAN ' + name + ' '), lambda: tensor)


def fix_r_index_keys(r_index):
    return {int(key): value for key, value in r_index.items()}


def tf_accuracy(predicted, target, mask=None, need_round=False, shape=None):
    with variable_scope('accuracy'):
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
        tensor_length = tf.shape(targets)[0]
        if need_round:
            predictions = tf.round(predictions)
            targets = tf.round(targets)
        predictions = tf.to_int32(predictions)
        targets = tf.to_int32(targets)

        eq_elements = tf.equal(predictions, targets)
        eq_elements = tf.to_int32(eq_elements)
        eq_count = tf.reduce_sum(eq_elements, axis=0)

        eq_count = tf.to_float(eq_count)
        length = tf.to_float(tensor_length)
        accuracy = eq_count / length

        real_accuracy = tf.cond(empty_tensor, lambda: 1.0, lambda: accuracy)

    return real_accuracy


def tf_length_mask(length):
    mask = tf.sequence_mask(length)
    mask = tf.transpose(mask, [1, 0])  # time major
    return mask


def tf_mask_gracefully(tensor, mask, sum_result=False):
    with variable_scope('mask_gracefully'):
        masked = tf.boolean_mask(tensor, mask)
        empty_tensor = tf.equal(tf.size(masked), 0)
        if sum_result:
            summed = tf.reduce_sum(masked)
            result = tf.cond(empty_tensor, lambda: 0.0, lambda: summed)
        else:
            mask_dims = len(mask.shape)
            tensor_dims = len(tensor.shape)
            result_dims = tensor_dims - mask_dims + 1
            result_shape = [1] * result_dims
            result = tf.cond(empty_tensor, lambda: tf.zeros(result_shape), lambda: masked)
    return result


def tf_conditional_lookup(condition, tensor, ids):
    with variable_scope('conditional_lookup'):
        zero_ids = tf.zeros_like(ids)
        lookup_ids = tf.where(condition, ids, zero_ids)
        tensor_values = tf.nn.embedding_lookup(tensor, lookup_ids)
        fallback = tf.zeros_like(tensor_values)
        _res_shape = tf.shape(tensor_values)
        _selector = tf.expand_dims(condition, axis=-1)
        selector = tf.broadcast_to(_selector, _res_shape)
        result = tf.where(selector, tensor_values, fallback)
    return result


def tf_conditional_ta_lookup(ta, condition, ids, fallback):
    with variable_scope('conditional_ta_lookup'):
        def selector(elem):
            cond, _id = elem
            return tf.cond(cond, lambda: ta.read(_id), lambda: fallback)

        res = tf.map_fn(selector, [condition, ids], dtype=tf.float32)
    return res
