import numpy as np
import tensorflow as tf
from .generator import Generator
from .discriminator import Discriminator

class CDCGAN(object):
    def __init__(self, d_params, g_params, zdim, ydim, xshape, lr=0.0002, beta1=0.5):
        self.is_training = tf.placeholder(tf.bool)

        self.zs = tf.placeholder(tf.float32, [None, zdim])
        self.g_ys = tf.placeholder(tf.float32, [None, ydim])

        self.xs = tf.placeholder(tf.float32, [None] + xshape)
        self.d_ys = tf.placeholder(tf.float32, [None, ydim])

        self.is_training = tf.placeholder(tf.bool)

        self.generator = Generator(g_params)
        self.discriminator = Discriminator(d_params)

        self.generator_output = self.generator(self.zs, self.g_ys, self.is_training)
        self.real_discriminator_output = self.discriminator(self.xs, self.d_ys, self.is_training)
        self.fake_discriminator_output = self.discriminator(self.generator_output, self.g_ys, self.is_training, reuse=True)

        self.generator_loss = -tf.reduce_mean(tf.log(self.fake_discriminator_output))
        self.discriminator_loss = -tf.reduce_mean(tf.log(self.real_discriminator_output) + tf.log(1.0 - self.fake_discriminator_output))

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        with tf.control_dependencies(g_update_ops):
            self.generator_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.generator_loss,
                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        with tf.control_dependencies(d_update_ops):
            self.discriminator_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.discriminator_loss,
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    def train_step(self, sess, xs, d_ys, zs, g_ys, is_training=True):
        _, dloss_curr = sess.run([self.discriminator_train_step, self.discriminator_loss],
                                    feed_dict={self.xs : xs, self.d_ys : d_ys, self.zs : zs, self.g_ys : d_ys, self.is_training : is_training})
        _, gloss_curr = sess.run([self.generator_train_step, self.generator_loss],
                                    feed_dict={self.zs : zs, self.g_ys : g_ys, self.is_training : is_training})
        return (gloss_curr, dloss_curr)

    def sample_generator(self, sess, zs, ys, is_training=True):
        return sess.run(self.generator_output, feed_dict={self.zs : zs, self.g_ys : ys, self.is_training : is_training})
