"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import tensorflow.contrib.layers as lays
from matplotlib import cm
import matplotlib
matplotlib.use('PDF')   # generate postscript output by default
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from dataset import *
from skimage import transform

# works with tensotflow 1.1.0.

tf.set_random_seed(1)

# Hyper Parameters
BATCHES = 8000
BATCH_SIZE = 64
PIXELS = 28        # Used to be 100.
LR = 0.002         # learning rate
NUM_TEST_IMG = 5
DRAW = False


def _plot(k, input_img, output_img, train_view_data, decoded_data):
    fig, a = plt.subplots(4, NUM_TEST_IMG, figsize=(10, 8))

    # original data (first row) for viewing
    for i in range(NUM_TEST_IMG):
        a[0][i].imshow(
            np.reshape(input_img[i], (PIXELS, PIXELS)))
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    # trained data
    for i in range(NUM_TEST_IMG):
        a[1][i].clear()
        a[1][i].imshow(output_img[i, ..., 0])
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    # original data (first row) for viewing
    for i in range(NUM_TEST_IMG):
        a[2][i].imshow(
            np.reshape(train_view_data[i], (PIXELS, PIXELS)))
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())

    # trained data
    for i in range(NUM_TEST_IMG):
        a[3][i].clear()
        a[3][i].imshow(decoded_data[i, ..., 0])
        a[3][i].set_xticks(())
        a[3][i].set_yticks(())

    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + os.sep + 'images_{:03}.pdf'.format(k))
    plt.close(fig)


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
f, a = plt.subplots(4, NUM_TEST_IMG, figsize=(8, 6))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = costmaps.test_targets[:NUM_TEST_IMG]
for i in range(NUM_TEST_IMG):
    a[0][i].imshow(
        np.reshape(view_data[i], (PIXELS, PIXELS)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

i = 0
test_view_data = costmaps.test_inputs[:NUM_TEST_IMG]
train_loss_ = 0.
test_loss_ = 0.
for step in range(BATCHES):
    b_x, b_y = costmaps.next_batch(BATCH_SIZE)
    _, decoded_, train_loss_ = sess.run(
        [train, decoded, loss],
        feed_dict={tf_x: resize_batch(b_x), tf_y: resize_batch(b_y)})
    if step % 10 == 0:  # plotting
        test_loss_ = sess.run(
            loss,
            {tf_x: resize_batch(costmaps.test_inputs[:100]),
             tf_y: resize_batch(costmaps.test_targets[:100])})
        print('step: {:8}, Epoch: {}, train loss: {:.4f}, test loss: {:.4f}'.format(
            step, costmaps.epochs_completed, train_loss_, test_loss_))
        # plotting decoded image (second row)
        prediction = sess.run(decoded, {tf_x: resize_batch(test_view_data)})
        _plot(i, b_x[:5], resize_batch(b_y[:5]),
              test_view_data, prediction)
        i += 1
