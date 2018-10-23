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
BATCH_SIZE = 64
PIXELS = 28        # Used to be 100.
LR = 0.002         # learning rate
NUM_TEST_IMG = 5


# Costmaps
costmaps = CostmapDataset(filename='costdata2d_10k.hdf5')
costmaps.reshape_data_to_tensors()

# plot one example
print(costmaps.train_inputs.shape)    # (55000, PIXELS * PIXELS)
print(costmaps.test_inputs.shape)     # (55000, 10)
plt.imshow(costmaps.test_targets[0].reshape((PIXELS, PIXELS)), cmap='gray')
plt.title('%i' % np.argmax('cost'))
plt.show()

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
f, a = plt.subplots(2, NUM_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = costmaps.test_targets[:NUM_TEST_IMG]
for i in range(NUM_TEST_IMG):
    a[0][i].imshow(
        np.reshape(view_data[i], (PIXELS, PIXELS)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in range(8000):
    b_y, b_x = costmaps.next_batch(BATCH_SIZE)
    _, encoded_, decoded_, loss_ = sess.run(
        [train, encoded, decoded, loss], {tf_x: b_x})

    if step % 100 == 0:     # plotting
        print('train loss: %.4f' % loss_)
        # plotting decoded image (second row)
        decoded_data = sess.run(decoded, {tf_x: view_data})
        for i in range(NUM_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(
                np.reshape(decoded_data[i], (PIXELS, PIXELS)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)
plt.ioff()
