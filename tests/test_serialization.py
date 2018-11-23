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

import __init__
from learning.dataset import *
from utils.misc import *
import numpy as np
from numpy.testing import assert_allclose
import pickle
import random
import time


def check_is_close(a, b, tolerance=1e-10):
    """ Returns True of all variable are close."""
    results = np.isclose(
        np.array(a),
        np.array(b),
        atol=tolerance)
    return results.all()


def test_hdf5_io():
    filename = "test_file.hdf5"
    A = np.random.random((1000, 1000))
    write_data_to_file(A, filename)
    B = load_data_from_file(filename)
    assert check_is_close(A, B)


def test_hdf5_dictionary_io():
    filename = "test_file.hdf5"
    dic_A = {}
    dic_A["first"] = np.array([4])
    dic_A["second"] = np.random.random(10)
    dic_A["third"] = np.random.random((10, 10))
    dic_A["forth"] = np.random.random((10, 10, 200))
    write_dictionary_to_file(dic_A, filename)
    dic_B = load_dictionary_from_file(filename)
    for key, value in list(dic_A.items()):
        assert check_is_close(dic_A[key], dic_B[key])
    # print dic_B["first"][0]
    # print dic_B["second"].shape
    object_B = dict_to_object(dic_B)
    assert object_B.first == dic_A["first"]


class TestObject:
    name = "blah"


def test_pickle_io():
    filename = 'test_file.pkl'
    thing1 = TestObject()
    with open(filename, 'wb') as file:
        print(thing1.name)
        pickle.dump(thing1, file)

    with open(filename, 'rb') as file:
        thing2 = pickle.load(file)
        print(thing2.name)

    os.remove('test_file.pkl')
    assert thing2.name == thing1.name


def test_paths_io():

    print("Test paths IO !")
    nb_env = 20
    nb_paths = 10

    paths = []
    for _ in range(nb_env):
        paths.append([])
        for _ in range(nb_paths):
            length = random.randint(10, 20)
            paths[-1].append(np.random.randint(50, size=(length, 2)))

    filename = "tmp_paths_{}.hdf5".format(str(time.time()).split('.')[0])
    print("save to tmp file : {}".format(filename))
    save_paths_to_file(paths, filename)
    saved_paths = load_paths_from_file(filename)
    saved_paths_file = learning_data_dir() + os.sep + filename
    assert os.path.isfile(saved_paths_file)
    os.remove(saved_paths_file)
    for i, k in product(list(range(nb_env)), list(range(nb_paths))):
        assert_allclose(paths[i][k], saved_paths[i][k])


if __name__ == "__main__":
    test_hdf5_io()
    test_hdf5_dictionary_io()
    test_pickle_io()
    test_paths_io()
