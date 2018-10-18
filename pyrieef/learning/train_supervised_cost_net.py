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


import optparse
from utils import *
from .dataset import *
from collections import namedtuple
from .networks import *
import time


def parse_options():

    parser = optparse.OptionParser("Supervised Cost Prediction")

    parser.add_option('--dataset', default="costdata2d_10k_small.hdf5",
                      type="string", help='path to training/test dataset')
    parser.add_option('--matlabdataset', action='store_true', default=False,
                      help='dataset is generated using MATLAB if this is set. Use MATIO to load it.')
    parser.add_option('--usecpu', action='store_true', default=False,
                      help='Do not use GPU, just use the CPU')
    parser.add_option('--gpu', default=1, type="int",
                      help='GPU to use (Default: 1)')
    parser.add_option('--saveDir', default='data', type="string",
                      help='directory to save/log experiments and trained models in')
    parser.add_option('--trainPer',  default=0.7,  type="float",
                      help='percentage of data to use for train/test')
    parser.add_option('--numEpochs', default=100, type="int",
                      help='number of epochs to run training/testing for')
    parser.add_option('--batchSize', default=1, type="int",
                      help='mini-batch size (1 = pure stochastic)')
    parser.add_option('--trainBPE', default=100, type="int",
                      help='number of batches of training per epoch')
    parser.add_option('--testBPE', default=50, type="int",
                      help='number of batches of testing per epoch')
    parser.add_option('--displayInterval', default=5, type="int",
                      help='number of training batches once to display')
    parser.add_option('--displayPort', default=8000, type="int",
                      help='port to draw images in. Localhost is the ip. Note that a server has to run for this port to show images')
    parser.add_option('--preTrained', action='store_true', default=False,
                      help='reload pretrained network and/or data (train/test files, seed, optim state, etc)')
    parser.add_option('--testPreTrained', action='store_true', default=False,
                      help='test a pre-trained network. No training')
    parser.add_option('--numTest', default=-1, type="int",
                      help='number of testing datasets (only used if testPreTrained is set)')
    parser.add_option('--loadOnlyModel', default=False, action='store_true',
                      help='if set, load only the pre-trained model. Do not load any other saved data')
    parser.add_option('--model', default='pred2dcost', type="string",
                      help='Type of model to train: pred2dcost | pred2dcostconv | pred2dcostfcn')
    parser.add_option('--usecudnn', action='store_true', default=False,
                      help='Use CUDNN convolutions/de-convolutions. Cannot be used in CPU mode')
    parser.add_option('--numskipadd', default=1, type="int",
                      help='Number of Skip+Add connections for the networks with an FCN structure')
    parser.add_option('--criterion', default='mse',
                      help='type of loss function/criterion to use to train: mse | abs')
    parser.add_option('--scaleCriterion', default=1, type="int",
                      help='scale factor to multiply the loss and/or any other penalties')
    parser.add_option('--noLossAveraging', default=False, action='store_true',
                      help='when set, loss is not averaged over all the data points')
    parser.add_option('--nonlinearity', default='prelu', type="string",
                      help='choose type of non-linearity the network uses: prelu | relu | tanh | sigmoid')
    parser.add_option('--seed', default=100, type="int",
                      help='seed for the random number generator. If < 0, chooses random seed')
    parser.add_option('--visualize', default=False, action='store_true',
                      help='visualize loss, input data, labels and predictions during training')
    parser.add_option('--visualizeWeights', default=False, action='store_true',
                      help='visualize layer weights during training')
    parser.add_option('--printGrads', default=False, action='store_true',
                      help='print the max/min gradients and the changes in weights at the end of each epoch')
    parser.add_option('--optimization', default='adam',
                      help='optimization method: sgd | adam')
    parser.add_option('--learningRate', default=1e-3, type="float",
                      help='learning rate at t = 0')
    parser.add_option('--learningRateDecay', default=1e-7, type="float",
                      help='learning rate decay - in terms of #training iterations (sgd only)')
    parser.add_option('--weightDecay', default=0, type="float",
                      help='weight decay (sgd only)')
    parser.add_option('--momentum', default=0, type="float",
                      help='momentum (sgd only)')
    parser.add_option('--L2', default=0, type="float",
                      help='coefficient for L2 regularization of the weights. If > 0 L2 regularization is enabled')
    parser.add_option('--batchNormalize', default=False, action='store_true',
                      help='Enable batch normalization across mini-batches')

    (opt, args) = parser.parse_args()
    print(opt)
    # if len(args) != 2:
    #     parser.error("incorrect number of arguments")
    return (opt, args)


def load_saved_network(savedFile, loadOnlyModel):
    """Load trained data """
    data = torch.load(savedFile)

    # Load all the data in the file
    if loadOnlyModel:
        print(('Loading only the saved network from: ', opt.preTrained))
        model = data.model  # Use cleaned up model
    else:
        print(('Loading all saved data from: ', opt.preTrained))
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


def create_loss_function(opt):
    """
     b1) Setup progress/loss logging (Do not overwrite existing
     files if pre-training an existing network)
        Setup loggers to save the training and test losses
    """
    if opt.model == 'pred2dcost':
        model = supervised2dcostprednet(
            opt.batchNormalize,
            opt.nonlinearity, opt.usecudnn)
    elif opt.model == 'pred2dcostconv':
        model = supervised2dcostprednet_onlyconv(
            opt.batchNormalize,
            opt.nonlinearity, opt.usecudnn)
    elif opt.model == 'pred2dcostfcn':
        model = supervised2dcostprednet_fcn(
            opt.batchNormalize,
            opt.nonlinearity,
            opt.usecudnn,
            opt.numskipadd)
    else:
        print(('Unknown model type input: ' + opt.model))
    print(('model : ' + str(model)))
    return model


def iterate(opt, dataset, num_iter, mode, total_iter):
    # Setup some vars
    loss_arr = None
    logger = None
    iter_counter = None
    if mode == 'Train':
        training()
        loss_arr = train_loss
        logger = train_loss_logger
        iter_counter = opt.trainIter
        totalIter = opt.numEpochs * opt.trainBPE
    elif mode == 'Test':
        # model:evaluate()
        loss_arr = test_loss
        logger = test_loss_logger
        iter_counter = opt.testIter
        totalIter = opt.numEpochs * opt.testBPE
    else:
        print(('Unknown mode: ' + mode))

if __name__ == '__main__':

    (opt, args) = parse_options()
    print((opt.preTrained))
    print((opt.dataset))
    data = CostmapDataset(opt.dataset)
    model = create_loss_function(opt)
