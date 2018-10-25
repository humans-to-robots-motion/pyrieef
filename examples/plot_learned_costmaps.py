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

import demos_common_imports
import tensorflow as tf
import tensorflow.contrib.layers as lays
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from pyrieef.learning.dataset import *
from skimage import transform

# works with tensotflow 1.1.0.

tf.set_random_seed(1)

# Hyper Parameters
BATCHES = 8000
BATCH_SIZE = 1000
PIXELS = 28        # Used to be 100.
LR = 0.002         # learning rate
NUM_TEST_IMG = 5
DRAW = False


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def autoencoder_cnn(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME',
                                activation_fn=tf.nn.sigmoid)
    return net

# Costmaps
costmaps = CostmapDataset(filename='costdata2d_55k.hdf5')
costmaps.normalize_maps()
costmaps.reshape_data_to_tensors()

# plot one example
print(costmaps.train_inputs.shape)    # (55000, PIXELS * PIXELS)
print(costmaps.test_inputs.shape)     # (55000, 10)
# plt.imshow(costmaps.test_targets[0].reshape((PIXELS, PIXELS)), cmap='gray')
# plt.title('%i' % np.argmax('cost'))
# plt.show()

# sys.exit(0)

tf_x = tf.placeholder(tf.float32, (None, 32, 32, 1))
tf_y = tf.placeholder(tf.float32, (None, 32, 32, 1))
decoded = autoencoder_cnn(tf_x)
# loss = tf.losses.mean_squared_error(
#     labels=tf_y,
#     predictions=decoded)
loss = tf.reduce_mean(tf.square(tf_y - decoded))
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# initialize figure
fig = plt.figure(figsize=(8, 4))
grid = plt.GridSpec(3, NUM_TEST_IMG, wspace=0.4, hspace=0.3)

a = [None] * 3
for i in range(3):
    a[i] = [None] * NUM_TEST_IMG
    for j in range(NUM_TEST_IMG):
        a[i][j] = fig.add_subplot(grid[i, j])
plt.ion()   # continuously plot

# original data (first row) for viewing
test_view_data_inputs = costmaps.test_inputs[:NUM_TEST_IMG]
test_view_data_targets = costmaps.test_targets[:NUM_TEST_IMG]
for i in range(NUM_TEST_IMG):
    a[0][i].imshow(
        np.reshape(test_view_data_inputs[i], (PIXELS, PIXELS)))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

    # original data (first row) for viewing
    a[1][i].imshow(
        np.reshape(test_view_data_targets[i], (PIXELS, PIXELS)))
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())

i = 0

train_loss_ = 0.
test_loss_ = 0.
# loss = []
for step in range(BATCHES):
    b_x, b_y = costmaps.next_batch(BATCH_SIZE)
    _, decoded_, train_loss_ = sess.run(
        [train, decoded, loss],
        feed_dict={tf_x: resize_batch(b_x), tf_y: resize_batch(b_y)})
    if step % 2 == 0:  # plotting
        test_loss_ = sess.run(
            loss,
            {tf_x: resize_batch(costmaps.test_inputs[:50]),
             tf_y: resize_batch(costmaps.test_targets[:50])})
        epoch = costmaps.epochs_completed
        infostr = str()
        infostr += 'step: {:8}, epoch: {:3}, '.format(step, epoch)
        infostr += 'train loss: {:.4f}, test loss: {:.4f}'.format(
            train_loss_, test_loss_)
        print(infostr)
        # loss.append([train_loss_, test_loss_])
        # plotting decoded image (second row)
        decoded_data = sess.run(
            decoded, {tf_x: resize_batch(test_view_data_inputs)})
        # trained data
        for i in range(NUM_TEST_IMG):
            a[2][i].clear()
            a[2][i].imshow(decoded_data[i, ..., 0])
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        i += 1
        plt.draw()
        plt.pause(0.01)

plt.ioff()
