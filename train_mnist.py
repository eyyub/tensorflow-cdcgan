import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model.cdcgan import CDCGAN

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
batch_size = 128
iters = 70000
draw_step = 500
device = '/gpu:0'

zdim = 100
ydim = 10

d_params = {
    'conv1' : {
        'filters' : [5, 5, 1, 32],
        'strides' : [1, 2, 2, 1],
        'padding' : 'SAME'
    },
    'conv2' : {
        'filters' : [5, 5, 32 + ydim, 16],
        'strides' : [1, 2, 2, 1],
        'padding' : 'SAME',
        'bn' : { 'center' : False, 'scale' : False }
    },
    'conv3' : {
        'filters' : [5, 5, 16, 8],
        'strides' : [1, 2, 2, 1],
        'padding' : 'SAME',
        'bn' : { 'center' : False, 'scale' : False }
    },
    'output' : {
        'wshape' : [4 * 4 * 8, 1]
    }
}

g_params = {
    'p' : {
        'wshape' : [zdim + ydim, 4 * 4 * 16],
        'bn' : {
            'center' : False,
            'scale' : False
        },
        'reshape' : [-1, 4, 4, 16]
    },
    'deconv1' : {
        'filters' : [5, 5, 32, 16],
        'strides' : [1, 2, 2, 1],
        'output' : [7, 7, 32], #batchdim
        'padding' : 'SAME',
        'bn' : { 'center' : False, 'scale' : False }
    },
    'deconv2' : {
        'filters' : [5, 5, 64, 32],
        'strides' : [1, 2, 2, 1],
        'output' : [14, 14, 64], #batchdim
        'padding' : 'SAME',
        'bn' : { 'center' : False, 'scale' : False }
    },
    'deconv3' : {
        'filters' : [5, 5, 1, 64],
        'strides' : [1, 2, 2, 1],
        'output' : [28, 28, 1], #batchdim
        'padding' : 'SAME'
    }
}

with tf.device(device):
    model = CDCGAN(d_params, g_params, zdim, ydim, [28, 28, 1])

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(iters):
        mini_batch_xs, mini_batch_ys = mnist.train.next_batch(batch_size)
        gloss, dloss = model.train_step(sess, np.reshape( 2. * mini_batch_xs - 1., [-1, 28, 28, 1]),
                    mini_batch_ys, np.random.uniform(-1, 1, (batch_size, zdim)), mini_batch_ys)
        if step % 500 == 1:
            imgs = model.sample_generator(sess, zs=np.repeat(np.random.uniform(-1, 1, (10, zdim)), 10, axis=0),
                                                ys=np.tile(np.eye(ydim), [10, 1]))
            fig = plt.figure()
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            for i in range(10*10):
                fig.add_subplot(10, 10, i + 1)
                plt.imshow(imgs[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig('images/iter_%d.jpg' % step)

            plt.close()
        print('Step %d : G loss %f | D loss %f' % (step, gloss, dloss))
