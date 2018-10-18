#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018

# An undercomplete occtomap to cost code
# ....

import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('PDF')   # generate postscript output by default
import matplotlib.pyplot as plt
from .visualize_data import *
from .dataset import *

num_epochs = 100   # Number of epochs to train the network
batch_size = 100   # Number of samples in each batch
batch_per_ep = 20
lr = 0.0001        # Learning rate


def _draw_row(fig, img1, img2, img3, i):
    draw_one_data_point(fig,
                        np.reshape(img1, (100, 100)),
                        np.reshape(img2, (100, 100)),
                        np.reshape(img3, (100, 100)),
                        4, i)


def _plot(ep, occ, cost_true, cost_pred):
    fig = plt.figure(figsize=(5, 6))
    _draw_row(fig, occ[0], cost_true[0], cost_pred[0], 0)
    _draw_row(fig, occ[1], cost_true[1], cost_pred[1], 1)
    _draw_row(fig, occ[2], cost_true[2], cost_pred[2], 2)
    _draw_row(fig, occ[3], cost_true[3], cost_pred[3], 3)

    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + os.sep + 'images_{:03}.pdf'.format(ep))


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
prediction = _autoencoder(x)

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
        test_loss_v, occ, cost_true, cost_pred = sess.run([
            loss, x, y, prediction])
        print(('Epoch: {}, Training Loss= {}, Test Loss= {}'.format(
            (ep + 1), train_loss_v, test_loss_v)))
        if ep % 10 == 0:
            _plot(ep, occ, cost_true, cost_pred)
