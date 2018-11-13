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

import opengl as gl
from skimage import data
from skimage.transform import rescale  # resize, downscale_local_mean
from skimage import img_as_ubyte
import warnings
import time
import matplotlib.pyplot as plt

use_matplotlib = False

image = data.chelsea()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # WARNING rescale
    image_scaled = img_as_ubyte(rescale(image, 1.0 / 4.0))

if not use_matplotlib:
    print((image_scaled.shape))
    viewer = gl.SimpleImageViewer()
    while True:
        viewer.imshow(image_scaled)
        time.sleep(0.01)
else:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[1].imshow(image_scaled)
    ax[1].set_title("Scaled image")
    plt.tight_layout()
    plt.show()
