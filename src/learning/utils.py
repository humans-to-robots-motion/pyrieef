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

from collections import namedtuple


def dict_to_object(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_object(v)
    return namedtuple('object', d.keys())(*d.values())


def load_saved_network(savedFile, loadOnlyModel):
    """Load trained data """
    data = torch.load(savedFile)

    # Load all the data in the file
    if loadOnlyModel:
        print('Loading only the saved network from: ', opt.preTrained)
        model = data.model  # Use cleaned up model
    else:
        print('Loading all saved data from: ', opt.preTrained)
        model = data.model  # Use cleaned up model
        rngstate = data.rngstate  # Random number generator state
        opt = data.opt or {}  # Save the options
        currEpoch = data.currepoch or 1  # Save the number of training epochs
        trainLoss = data.trainloss or {}  # Training loss/batch
        testLoss = data.testloss or {}  # Test loss/batch
        optimMethod = data.optimmethod or nil  # Save optimization method
        # Save optimizer state (for optimizers like ADAM, this helps)
        optimState = data.optimstate or nil
        torch.setRNGState(data.rngstate)  # Set random number state
