import numpy as np
import tensorflow as tf
from .utils import get_shape, batch_normalization

class Generator(object):
    def __init__(self, arch_params, stddev=0.02):
        self.stddev = stddev
        self.arch_params = arch_params

    def __call__(self, zs, ys, is_training):
        batch_dim = tf.shape(zs)[0]
        with tf.variable_scope('generator', initializer=tf.truncated_normal_initializer(stddev=self.stddev)):
            inputs = tf.concat([zs, ys], axis=1)
            with tf.variable_scope('p'): # Project and reshape
                v = self.arch_params['p']
                W_p = tf.get_variable('W', v['wshape'])
                z_p = tf.matmul(inputs, W_p)
                bn_p = batch_normalization(z_p, center=v['bn']['center'],
                                            scale=v['bn']['scale'],
                                            training=is_training) if 'bn' in v else z_p
                a_p = tf.nn.relu(bn_p)
                reshaped_a_p = tf.reshape(a_p, v['reshape'])

            with tf.variable_scope('deconv1'): # deconvolution here means conv transpose
                v = self.arch_params['deconv1']
                filters_1 = tf.get_variable('filters', v['filters'])
                deconv_1 = tf.nn.conv2d_transpose(reshaped_a_p, filters_1,
                                                    [batch_dim] + v['output'],
                                                    v['strides'], padding=v['padding'])
                bn_1 = batch_normalization(tf.reshape(deconv_1, [batch_dim] + v['output']),
                                            center=v['bn']['center'],
                                            scale=v['bn']['scale'],
                                            training=is_training) if 'bn' in v else deconv_1
                a_1 = tf.nn.relu(bn_1)

            with tf.variable_scope('deconv2'): # deconvolution here means conv transpose
                v = self.arch_params['deconv2']
                filters_2 = tf.get_variable('filters', v['filters'])
                deconv_2 = tf.nn.conv2d_transpose(a_1, filters_2,
                                                    [batch_dim] + v['output'],
                                                    v['strides'], padding=v['padding'])
                bn_2 = batch_normalization(tf.reshape(deconv_2, [batch_dim] + v['output']),
                                            center=v['bn']['center'],
                                            scale=v['bn']['scale'],
                                            training=is_training) if 'bn' in v else deconv_2
                a_2 = tf.nn.relu(bn_2)

            with tf.variable_scope('deconv3'): # deconvolution here means conv transpose
                v = self.arch_params['deconv3']
                filters_3 = tf.get_variable('filters', v['filters'])
                deconv_3 = tf.nn.conv2d_transpose(a_2, filters_3, [batch_dim] + v['output'],
                                                    v['strides'], padding=v['padding'])
                bn_3 = batch_normalization(tf.reshape(deconv_3, [batch_dim] + v['output']),
                                            center=v['bn']['center'],
                                            scale=v['bn']['scale'],
                                            training=is_training) if 'bn' in v else deconv_3
                a_3 = tf.nn.tanh(bn_3)
        return a_3
