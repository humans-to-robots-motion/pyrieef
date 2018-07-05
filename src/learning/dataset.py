#!/usr/bin/env python

# Copyright (c) 2018 University of Stuttgart
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


import h5py
import os
from utils import *
import numpy as np
# TODO write some import test


class CostmapDataset:

    def __init__(self, filename):
        print('==> Loading dataset from: ' + filename)
        data = dict_to_object(load_dictionary_from_file(filename))
        self._size_limit = True
        self._max_index = 1000
        self.inputs = []
        self.targets = []
        print('Sorting out inputs and targets...')
        for i, d in enumerate(data.datasets):
            self.inputs.append(d[0])  # Occupancy maps
            self.targets.append(d[2])  # Costmaps
            # show_progress(i, len(data.datasets))
            if i == self._max_index - 1 and self._size_limit:
                break
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
        print(' - num. inputs : {}, shape : {}'.format(
            len(self.inputs),
            self.inputs.shape))
        print(' - num. targets : {}, shape : {}'.format(
            len(self.targets),
            self.targets.shape))


def learning_data_dir():
    return os.path.abspath(os.path.dirname(__file__)) + os.sep + "data"


def write_data_to_file(data_out, filename='costdata2d_10k.hdf5'):
    directory = learning_data_dir()
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = h5py.File(directory + os.sep + filename, 'w')
    f.create_dataset("mydataset", data=data_out)
    f.close()


def load_data_from_file(filename='costdata2d_10k.hdf5'):
    with h5py.File(learning_data_dir() + os.sep + filename, 'r') as f:
        datasets = f['mydataset'][:]
    return datasets


def write_dictionary_to_file(data_out, filename='costdata2d_10k.hdf5'):
    directory = learning_data_dir()
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = h5py.File(directory + os.sep + filename, 'w')
    for key, value in data_out.items():
        f.create_dataset(key, data=value)
    f.close()


def load_dictionary_from_file(filename='costdata2d_10k.hdf5'):
    datasets = {}
    with h5py.File(learning_data_dir() + os.sep + filename, 'r') as f:
        for d in f:
            # print("d : " + d)
            print("f[d] : " + str(d))
            datasets[str(d)] = f[d][:]
    return datasets


def load_data_from_hdf5(filename, train_per):
    """ Setup training / test data  """
    # Depracted
    print('==> Loading dataset from: ' + filename)
    data = dict_to_object(load_dictionary_from_file(filename))
    print('==> Finished loading data')
    image_height = data.size[0]
    image_width = data.size[1]
    train_data = []
    test_data = []
    train_data_ids = []
    test_data_ids = []
    if train_data_ids and test_data_ids:
        print "We have some data ids"
        numTrain = len(train_data_ids)
        numTest = len(test_data_ids)
        numData = num_train + num_test
        for k in range(numTrain):
            train_data.append(data.datasets[train_data_ids[k]])
        for k in range(numTrain):
            test_data.append(data.datasets[test_data_ids[k]])
    else:
        # Load datasets afresh
        num_data = len(data.datasets)  # Total number of datasets
        num_train = int(round(train_per * num_data))
        num_test = num_data - num_train
        for k in range(num_data):
            if k < num_train:
                train_data.append(data.datasets[k])
                train_data_ids.append(k)
            else:
                test_data.append(data.datasets[k])
                test_data_ids.append(k)

        assert len(train_data) == num_train and len(test_data) == num_test

    print('Num. total: {}, Num. train: {}; Num. test: {}'.format(
        num_data, num_train, num_test))
    return train_data, test_data


def import_tf_data(filename='costdata2d_10k.hdf5'):
    import tensorflow as tf
    rawdata = CostmapDataset(filename)
    # Assume that each row of
    # `inputs` corresponds to the same row as `targets`.
    assert rawdata.inputs.shape[0] == rawdata.targets.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((
        rawdata.inputs,
        rawdata.targets))
    print(dataset.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset.output_shapes)  # ==> "(10, ((), (100,)))"
    return dataset