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
            # print ("f[d] : " + str(f[d][:]))
            datasets[str(d)] = f[d][:]
    return datasets
