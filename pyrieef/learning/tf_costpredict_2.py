"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
from dataset import *

# works with tensotflow 1.1.0.

tf.set_random_seed(1)

# Hyper Parameters
BATCHES = 8000
BATCH_SIZE = 64
PIXELS = 28        # Used to be 100.
LR = 0.002         # learning rate
NUM_TEST_IMG = 5
DRAW = False


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

# tf placeholder
# value in the range of (0, 1)
tf_x = tf.placeholder(tf.float32, [None, PIXELS * PIXELS])

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encoded = tf.layers.dense(en2, 3)

# decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de2, PIXELS * PIXELS, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# initialize figure
f, a = plt.subplots(4, NUM_TEST_IMG, figsize=(8, 6))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = costmaps.test_targets[:NUM_TEST_IMG]
for i in range(NUM_TEST_IMG):
    a[0][i].imshow(
        np.reshape(view_data[i], (PIXELS, PIXELS)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# original data (first row) for viewing
train_view_data = costmaps.train_targets[:NUM_TEST_IMG]
for i in range(NUM_TEST_IMG):
    a[2][i].imshow(
        np.reshape(train_view_data[i], (PIXELS, PIXELS)), cmap='gray')
    a[2][i].set_xticks(())
    a[2][i].set_yticks(())

for step in range(BATCHES):
    b_x, b_y = costmaps.next_batch(BATCH_SIZE)
    _, encoded_, decoded_, train_loss_ = sess.run(
        [train, encoded, decoded, loss], {tf_x: b_y})

    if step % 100 == 0:  # plotting
        test_loss_ = sess.run(loss, {tf_x: costmaps.test_targets})
        print('Epoch: {}, train loss: {:.4f}, test loss: {:.4f}'.format(
            costmaps.epochs_completed,
            train_loss_, test_loss_))
        # plotting decoded image (second row)
        decoded_data = sess.run(decoded, {tf_x: view_data})
        for i in range(NUM_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(
                np.reshape(decoded_data[i], (PIXELS, PIXELS)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        decoded_data = sess.run(decoded, {tf_x: train_view_data})
        for i in range(NUM_TEST_IMG):
            a[3][i].clear()
            a[3][i].imshow(
                np.reshape(decoded_data[i], (PIXELS, PIXELS)), cmap='gray')
            a[3][i].set_xticks(())
            a[3][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)
plt.ioff()
