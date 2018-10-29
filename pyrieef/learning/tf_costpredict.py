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
from visualize_data import *
from dataset import *

NUM_EPOCHS = 100            # Number of epochs to train the network
BATCH_SIZE = 100            # Number of samples in each batch
BATCH_PER_EP = 20
LEARNING_RATE = 0.0001      # Learning rate
PIXELS = 100                # Used to be 100.
DRAW_EPOCH = True


def _draw_row(fig, img1, img2, img3, i):
    limits = np.array([[0, 1], [0, 1]])  # x_min, x_max, y_min, y_max
    draw_one_data_point(fig,
                        limits,
                        np.reshape(img1, (PIXELS, PIXELS)),
                        np.reshape(img2, (PIXELS, PIXELS)),
                        np.reshape(img3, (PIXELS, PIXELS)),
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
    plt.close(fig)


def _preprocess(x, y):
    x = tf.reshape(x, [PIXELS, PIXELS, 1])
    y = tf.reshape(y, [PIXELS, PIXELS, 1])
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
    net = lays.fully_connected(net, 100)
    # decoder
    # 1 x 1 x 128    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  100 x 100 x 1
    net = lays.conv2d_transpose(net, 64,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 32,    [5, 5], stride=2)
    net = lays.conv2d_transpose(net, 16,    [5, 5], stride=5)
    net = lays.conv2d_transpose(net, 1,     [5, 5], stride=5)
    return net


# read costmap dataset
train_ds, test_ds = import_tf_data()
test_ds = test_ds.map(_preprocess)
test_ds = test_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(_preprocess)
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.repeat(NUM_EPOCHS)
iterator = tf.data.Iterator.from_structure(
    train_ds.output_types,
    train_ds.output_shapes)
x, y = iterator.get_next()
train_init_op = iterator.make_initializer(train_ds)
test_init_op = iterator.make_initializer(test_ds)

# calculate the loss and optimize the network
# claculate the mean square error loss
loss = tf.losses.mean_squared_error(_autoencoder(x), y)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
prediction = _autoencoder(x)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Compute for 100 epochs.
    for ep in range(NUM_EPOCHS):
        sess.run(train_init_op)
        for i in range(BATCH_PER_EP):
            _, train_loss_v = sess.run([train_op, loss])
            # print('Epoch: {}, Training Loss= {}'.format((ep + 1), loss_value))

        sess.run(test_init_op)
        test_loss_v, occ, cost_true, cost_pred = sess.run([
            loss, x, y, prediction])
        print(('Epoch: {}, Training Loss= {}, Test Loss= {}'.format(
            (ep + 1), train_loss_v, test_loss_v)))

        if ep == 0:
            cost_pred_prev = cost_pred

        if DRAW_EPOCH and (ep % 1 == 0):
            _plot(ep, occ, cost_true, cost_pred)

            # print(" -- Diff : ", np.linalg.norm(
            #     cost_pred - cost_pred_prev, axis=1).sum())
            # print(" -- shape : ", cost_pred_prev.shape)
            cost_pred_prev = cost_pred
