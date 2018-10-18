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

import torch
import tensorflow


def load_criterion(criType, averageLoss):
    """  Code to load a criterion """
    criterion
    if criType == 'mse':
        criterion = nn.MSECriterion(averageLoss)
    elif criType == 'wtmse':
        # Empty weight tensor
        criterion = nn.ScaledMSECriterion(torch.Tensor(), averageLoss)
    elif criType == 'abs':
        criterion = nn.AbsCriterion(averageLoss)
    else:
        print(('Unknown criterion type input: ' + criType))
    return criterion


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(
        net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net


def supervised2dcostprednet(
        _batchNormalize, _nonlinearity, _usecudnn):
    model = None
    return model


def supervised2dcostprednet_onlyconv(
        _batchNormalize, _nonlinearity, _usecudnn):
    model = None
    return model


def supervised2dcostprednet_fcn(
        _batchNormalize, _nonlinearity, _usecudnn, _numskipadd):
    model = None
    return model


def supervised2dcostprednet_linear(
        _batchNormalize, _nonlinearity, _usecudnn, _numskipadd):
    model = None
    return model
