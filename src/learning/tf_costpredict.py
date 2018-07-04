#!/usr/bin/env python
# An undercomplete occtomap to cost code
# ....
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import *

batch_size = 100  # Number of samples in each batch
num_epochs = 10   # Number of epochs to train the network
lr = 0.001        # Learning rate


def _process(x, y):
    x = tf.reshape(x, [100, 100, 1])
    y = tf.reshape(y, [100, 100, 1])
    return x, y


def _autoencoder(inputs):
    # encoder
    # 100 x 100 x 1  ->  n x n x 32
    # n x n x 32     ->  8 x 8 x 64
    # 8 x 8 x 16     ->  2 x 2 x 128
    net = lays.conv2d(inputs,   16,   [5, 5], stride=4)
    net = lays.conv2d(net,      32,   [5, 5], stride=4)
    net = lays.conv2d(net,      64,   [5, 5], stride=2)
    net = lays.conv2d(net,      128,  [5, 5], stride=2)
    net = lays.conv2d(net,      256,  [5, 5], stride=2)
    # fullyconnected
    # print(str(net))
    net = lays.fully_connected(net, 100)
    # decoder
    # 1 x 1 x 128    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  100 x 100 x 1
    net = lays.conv2d_transpose(net, 50,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 25,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 10,    [5, 5], stride=5)
    net = lays.conv2d_transpose(net, 1,     [5, 5], stride=5,
                                activation_fn=tf.nn.tanh)
    return net

x = tf.placeholder(tf.float64, (None, 100, 100, 1))
y = _autoencoder(x)

# calculate the loss and optimize the network
# claculate the mean square error loss
loss = tf.reduce_mean(tf.square(x - y))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# read costmap dataset
dataset = import_tf_data()
dataset = dataset.map(_process)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
batch_per_ep = int(1000 / batch_size)

# initialize the network
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Compute for 100 epochs.
    for ep in range(num_epochs):
        sess.run(iterator.initializer)
        for i in range(batch_per_ep):
            _, c = sess.run([train_op, loss],
                            feed_dict={x: next_element[0].eval(),
                                       y: next_element[1].eval()})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
