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
from pyrieef.learning.dataset import *
from pyrieef.learning.tf_networks import *

# works with tensotflow 1.1.0.

# Hyper Parameters
BATCHES = 8000
BATCH_SIZE = 64
PIXELS = 28        # Used to be 100.
LR = 0.002         # learning rate
NUM_TEST_IMG = 5
DRAW = False
SAVE_TO_FILE = True
if DRAW or SAVE_TO_FILE:
    import matplotlib
    if SAVE_TO_FILE:
        matplotlib.use('PDF')   # generate postscript output by default
    import matplotlib.pyplot as plt

tf.set_random_seed(1)

# Define Network
# network = ConvDeconv64()
network = ConvDeconvSmall()


# Costmaps
costmaps = CostmapDataset(filename='costdata2d_55k.hdf5')
costmaps.normalize_maps()
costmaps.reshape_data_to_tensors()

tf_x = network.placeholder()
tf_y = network.placeholder()
decoded = network.define(tf_x)
# loss = tf.losses.mean_squared_error(
#     labels=tf_y,
#     predictions=decoded)
loss = tf.reduce_mean(tf.square(tf_y - decoded))
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# initialize figure
if DRAW or SAVE_TO_FILE:
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
        a[0][i].imshow(test_view_data_inputs[i].reshape(28, 28))
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

        # original data (first row) for viewing
        a[1][i].imshow(test_view_data_targets[i].reshape(28, 28))
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

k = 0

train_loss_ = 0.
test_loss_ = 0.
# loss = []
for step in range(BATCHES):
    b_x, b_y = costmaps.next_batch(BATCH_SIZE)
    _, decoded_, train_loss_ = sess.run(
        [train, decoded, loss],
        feed_dict={tf_x: network.resize_batch(b_x),
                   tf_y: network.resize_batch(b_y)})
    if step % 2 == 0:  # plotting
        test_loss_ = sess.run(
            loss,
            {tf_x: network.resize_batch(costmaps.test_inputs[:50]),
             tf_y: network.resize_batch(costmaps.test_targets[:50])})
        epoch = costmaps.epochs_completed
        infostr = str()
        infostr += 'step: {:8}, epoch: {:3}, '.format(step, epoch)
        infostr += 'train loss: {:.4f}, test loss: {:.4f}'.format(
            train_loss_, test_loss_)
        print(infostr)
        # loss.append([train_loss_, test_loss_])
        # plotting decoded image (second row)

        if DRAW or SAVE_TO_FILE:
            decoded_data = sess.run(
                decoded, {tf_x: network.resize_batch(test_view_data_inputs)})
            for i in range(NUM_TEST_IMG):
                a[2][i].clear()
                a[2][i].imshow(network.resize_output(decoded_data, i))
                a[2][i].set_xticks(())
                a[2][i].set_yticks(())
            if SAVE_TO_FILE and (k % 20 == 0):
                directory = learning_data_dir() + os.sep + "results"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig.savefig(directory + os.sep + 'images_{:03}.pdf'.format(k))
                plt.close(fig)
            k += 1
            if DRAW:
                plt.draw()
                plt.pause(0.01)

plt.ioff()
