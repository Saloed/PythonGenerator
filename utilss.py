import _pickle as P
import tensorflow as tf


def dump_object(obj, name):
    with open(name, 'wb') as f:
        P.dump(obj, f)


def load_object(name):
    with open(name, 'rb') as f:
        return P.load(f)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_object(_dict):
    return Struct(**_dict)


def print_shape(tensor, name):
    return tf.Print(tensor, [tf.shape(tensor)], name + ' ')
