#!/usr/bin/env python
# An undercomplete occtomap to cost code
# ....
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import *

num_epochs = 100   # Number of epochs to train the network
batch_size = 100   # Number of samples in each batch
batch_per_ep = 20
lr = 0.0001        # Learning rate


def _preprocess(x, y):
    x = tf.reshape(x, [100, 100, 1])
    y = tf.reshape(y, [100, 100, 1])
    return x, y


def _autoencoder(inputs):
    # encoder
    # 100 x 100 x 1  ->  n x n x 32
    # n x n x 32     ->  8 x 8 x 64
    # 8 x 8 x 16     ->  2 x 2 x 128
    net = lays.conv2d(inputs,   16,   [5, 5], stride=5)
    net = lays.conv2d(net,      32,   [5, 5], stride=5)
    net = lays.conv2d(net,      64,   [5, 5], stride=2)
    net = lays.conv2d(net,      128,  [5, 5], stride=2)
    # net = lays.conv2d(net,      256,  [5, 5], stride=2)
    # fullyconnected
    # print(str(net))
    # net = lays.fully_connected(net, 100)
    # decoder
    # 1 x 1 x 128    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  100 x 100 x 1
    net = lays.conv2d_transpose(net, 64,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 32,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 16,    [5, 5], stride=5)
    net = lays.conv2d_transpose(net, 1,     [5, 5], stride=5,
                                activation_fn=tf.nn.tanh)
    return net


# read costmap dataset
train_ds, test_ds = import_tf_data()
test_ds = test_ds.map(_preprocess)
test_ds = test_ds.batch(batch_size)
train_ds = train_ds.map(_preprocess)
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.repeat(num_epochs)
iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                           train_ds.output_shapes)
x, y = iterator.get_next()
train_init_op = iterator.make_initializer(train_ds)
test_init_op = iterator.make_initializer(test_ds)

# calculate the loss and optimize the network
# claculate the mean square error loss
loss = tf.losses.mean_squared_error(_autoencoder(x), y)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Compute for 100 epochs.
    for ep in range(num_epochs):
        sess.run(train_init_op)
        for i in range(batch_per_ep):
            _, train_loss_v = sess.run([train_op, loss])
            # print('Epoch: {}, Training Loss= {}'.format((ep + 1), loss_value))
        sess.run(test_init_op)
        test_loss_v = sess.run(loss)
        print('Epoch: {}, Training Loss= {}, Test Loss= {}'.format(
            (ep + 1), train_loss_v, test_loss_v))
