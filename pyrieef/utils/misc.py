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

# from collections import namedtuple
import sys
import os
from time import sleep
import math
import operator


def row_major(i, dim):
    """
    Returns coordinates in a matrix with row major indexing
    """
    return int(i % dim), int(math.floor(i / dim))


def dict_to_object(d):
    """
    Converts a dictionary to an object

        Use to work with
        namedtuple('object', d.keys())(*d.values())

    TODO: bit sure what the first part is for.
    """
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = dict_to_object(v)

    class DObj(object):
        pass
    dobj = DObj()
    dobj.__dict__ = d
    return dobj


def show_progress(idx, idx_max):
    sys.stdout.flush()
    msg = "progress %i %%" % (100. * float(idx) / float(idx_max))
    sys.stdout.write(msg + chr(8) * len(msg))
    if idx >= idx_max:
        sys.stdout.flush()
        sys.stdout.write("DONE" + " " * len(msg) + "\n")
    sleep(0.0001)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def pad_zeros(string, number, maximum):
    zeros = math.ceil(math.log10(maximum))
    return string + '{:0{width}}'.format(number, width=str(int(zeros)))


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))

# ____________________________________________________________________________
# Grid Functions


orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)
