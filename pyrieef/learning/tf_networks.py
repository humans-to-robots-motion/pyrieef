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

    def resize_output(self, imgs, i):
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


class ConvDeconvResize(Network):

    """
    https://towardsdatascience.com/
    autoencoders-introduction-and-implementation-3f40483b0a85
    """

    def __init__(self):
        return

    def placeholder(self):
        # return = tf.placeholder(tf.float32, (None, 28,28,1), name="input")
        return tf.placeholder(tf.float32, (None, 28, 28, 1))

    def resize_batch(self, imgs):
        return imgs.reshape((-1, 28, 28, 1))

    def resize_output(self, imgs, i):
        return imgs[i, ..., 0]

    def define(self, x_inputs):

        print("---------------------------------------------")
        print(" Define layers of auto encoder !!!")
        print("---------------------------------------------")

        # Encoder
        conv1 = tf.layers.conv2d(
            inputs=x_inputs, filters=16, kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv1.get_shape())
        # Now 28x28x16
        maxpool1 = tf.layers.max_pooling2d(
            conv1, pool_size=(2, 2), strides=(2, 2),
            padding='same')
        print(maxpool1.get_shape())
        # Now 14x14x16
        conv2 = tf.layers.conv2d(
            inputs=maxpool1, filters=8, kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv2.get_shape())
        # Now 14x14x8
        maxpool2 = tf.layers.max_pooling2d(
            conv2, pool_size=(2, 2), strides=(2, 2),
            padding='same')
        print(maxpool2.get_shape())
        # Now 7x7x8
        conv3 = tf.layers.conv2d(
            inputs=maxpool2, filters=8, kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv3.get_shape())
        # Now 7x7x8
        encoded = tf.layers.max_pooling2d(
            conv3, pool_size=(2, 2), strides=(2, 2),
            padding='same')
        print(encoded.get_shape())
        # Now 4x4x8

        # Decoder
        upsample1 = tf.image.resize_images(
            encoded, size=(7, 7),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        print(upsample1.get_shape())
        # Now 7x7x8
        conv4 = tf.layers.conv2d(
            inputs=upsample1, filters=8, kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv4.get_shape())
        # Now 7x7x8
        upsample2 = tf.image.resize_images(
            conv4, size=(14, 14),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        print(upsample2.get_shape())
        # Now 14x14x8
        conv5 = tf.layers.conv2d(
            inputs=upsample2, filters=8, kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv5.get_shape())
        # Now 14x14x8
        upsample3 = tf.image.resize_images(
            conv5, size=(28, 28),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        print(upsample3.get_shape())
        # Now 28x28x8
        conv6 = tf.layers.conv2d(
            inputs=upsample3, filters=16,
            kernel_size=(3, 3),
            padding='same', activation=tf.nn.relu)
        print(conv6.get_shape())
        # Now 28x28x16

        logits = tf.layers.conv2d(
            inputs=conv6, filters=1,
            kernel_size=(3, 3),
            padding='same', activation=None)
        print(logits.get_shape())
        # Now 28x28x1

        # Pass logits through sigmoid to get reconstructed image
        decoded = tf.nn.sigmoid(logits)
        print(decoded.get_shape())

        self._network = decoded
        print("total number of parameters : ", self.number_of_parameters())
        return decoded
