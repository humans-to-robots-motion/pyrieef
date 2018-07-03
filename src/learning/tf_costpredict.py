#!/usr/bin/env python
# An undercomplete occtomap to cost code
# ....
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import *

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate


def autoencoder(inputs):
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
    net = lays.fully_connected(net, 100)
    net = lays.fully_connected(net, 50)
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

# read costmap dataset
dataset = import_tf_data()

# Define training and validation datasets with the same structure.
training_dataset = dataset.range(100)
# validation_dataset = dataset.range(50)

print(training_dataset.output_types)
print(training_dataset.output_shapes)
# calculate the number of batches per epoch
# batch_per_ep = training_dataset // batch_size

iterator = tf.data.Iterator.from_structure(
    training_dataset.output_types,
    training_dataset.output_shapes)

ae_inputs = tf.placeholder(tf.float32, (None, 100, 100, 1))
ae_outputs = autoencoder(ae_inputs)

# calculate the loss and optimize the network
# claculate the mean square error loss
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
