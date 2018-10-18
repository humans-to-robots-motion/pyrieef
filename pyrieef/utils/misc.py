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

from collections import namedtuple
import sys
import os
from time import sleep


def dict_to_object(d):
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = dict_to_object(v)
    return namedtuple('object', list(d.keys()))(*list(d.values()))


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
