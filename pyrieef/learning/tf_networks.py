import tensorflow as tf
import tensorflow.contrib.layers as lays
import numpy as np
from skimage import transform


class Network:

    """ Convenience class to easily swap networks """

    def __init__(self):
        self._network = None
        return

    def placeholder(self):
        return NotImplementedError()

    def define(self):
        return NotImplementedError()

    def resize_batch(self, b):
        return NotImplementedError()

    def number_of_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


class ConvDeconvSmall(Network):

    """
    Defines a Convolution Deconvolution network
    sized for the Mnist 28x28 matrix format
    """

    def __init__(self):
        self._l = 28

    def input(self):
        return self.tf_x

    def placeholder(self):
        return tf.placeholder(tf.float32, (None, 32, 32, 1))

    def resize_batch(self, imgs):
        imgs = imgs.reshape((-1, 28, 28, 1))
        resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
        for i in range(imgs.shape[0]):
            resized_imgs[i, ..., 0] = transform.resize(
                imgs[i, ..., 0], (32, 32))
        return resized_imgs

    def resize_output(self, imgs, i):
        return imgs[i, ..., 0]

    def define(self, x_input):
        print("---------------------------------------------")
        print(" Define layers of auto encoder !!!")
        print("---------------------------------------------")
        # encoder
        # 32 x 32 x 1   ->  16 x 16 x 32
        # 16 x 16 x 32  ->  8 x 8 x 16
        # 8 x 8 x 16    ->  2 x 2 x 8
        net = lays.conv2d(x_input, 32, [5, 5], stride=2, padding='SAME')
        print(net.get_shape())
        net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
        print(net.get_shape())
        net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
        print(net.get_shape())
        # decoder
        # 2 x 2 x 8    ->  8 x 8 x 16
        # 8 x 8 x 16   ->  16 x 16 x 32
        # 16 x 16 x 32  ->  32 x 32 x 1
        net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
        print(net.get_shape())
        net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
        print(net.get_shape())
        net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME',
                                    activation_fn=tf.nn.sigmoid)
        print(net.get_shape())
        self._network = net
        print("total number of parameters : ", self.number_of_parameters())
        return self._network


class ConvDeconv64(Network):

    """
    Defines a Convolution Deconvolution network
    sized for the Mnist 28x28 matrix format
    """

    def __init__(self):
        return

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def placeholder(self):
        return tf.placeholder(tf.float32, (None, 28, 28))

    def resize_batch(self, imgs):
        return imgs.reshape((-1, 28, 28))

    def resize_output(self, imgs, i):
        return imgs[i]

    def define(self, x_input):

        print("---------------------------------------------")
        print(" Define layers of auto encoder !!!")
        print("---------------------------------------------")

        dec_in_channels = 1
        n_latent = 8

        reshaped_dim = [-1, 7, 7, dec_in_channels]
        inputs_decoder = 49 * dec_in_channels / 2
        activation = self.lrelu
        nb_filters = 64

        # Encoder.
        X = tf.reshape(x_input, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(
            X, filters=nb_filters, kernel_size=4, strides=2,
            padding='same', activation=activation)
        print(x.get_shape())
        x = tf.layers.conv2d(
            x, filters=nb_filters, kernel_size=4, strides=2,
            padding='same', activation=activation)
        print(x.get_shape())
        x = tf.layers.conv2d(
            x, filters=nb_filters, kernel_size=4, strides=1,
            padding='same', activation=activation)
        print(x.get_shape())
        x = tf.contrib.layers.flatten(x)
        print(x.get_shape())
        x = tf.layers.dense(x, units=n_latent)
        print(x.get_shape())

        # Decoder.
        x = tf.layers.dense(
            x, units=inputs_decoder * 2 + 1,
            activation=self.lrelu)
        print(x.get_shape())
        x = tf.reshape(x, reshaped_dim)
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(
            x, filters=nb_filters, kernel_size=4, strides=2,
            padding='same', activation=tf.nn.relu)
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(
            x, filters=nb_filters, kernel_size=4, strides=1,
            padding='same', activation=tf.nn.relu)
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(
            x, filters=nb_filters, kernel_size=4, strides=1,
            padding='same', activation=tf.nn.relu)
        print(x.get_shape())

        x = tf.contrib.layers.flatten(x)
        print(x.get_shape())
        x = tf.layers.dense(
            x, units=28 * 28, activation=tf.nn.sigmoid)
        print(x.get_shape())
        img = tf.reshape(x, shape=[-1, 28, 28])

        self._network = img
        print("total number of parameters : ", self.number_of_parameters())
        return img
