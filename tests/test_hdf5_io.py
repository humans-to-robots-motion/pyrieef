#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
#                                        Jim Mainprice on Sunday June 17 2017
import __init__
from learning.dataset import *
from learning.utils import *
import numpy as np


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
    for key, value in dic_A.items():
        assert check_is_close(dic_A[key], dic_B[key])
    # print dic_B["first"][0]
    # print dic_B["second"].shape
    object_B = dict_to_object(dic_B)
    assert object_B.first == dic_A["first"]


if __name__ == "__main__":
    test_hdf5_io()
    test_hdf5_dictionary_io()
