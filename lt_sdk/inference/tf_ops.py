import os

import tensorflow as tf


def load_ops():
    return tf.load_op_library(os.path.join(os.path.dirname(__file__), "lib_tf_ops.so"))
