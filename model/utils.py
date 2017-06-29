import tensorflow as tf

def get_shape(tensor): # static shape
    return tensor.get_shape().as_list()

def batch_normalization(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn
