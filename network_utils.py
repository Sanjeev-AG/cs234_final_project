"""
This file contains utility functions for constructing and working with neural networks.
"""

import numpy as np
import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    """
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    It should have 'n_layers' layers, each of 'size' units and followed
    by a ReLU nonlinearity. Additionally, the final layer should be linear (no ReLU).

    That is, the network architecture should be the following:
    [LINEAR LAYER]_1 -> [RELU] -> [LINEAR LAYER]_2 -> ... -> [LINEAR LAYER]_n -> [RELU] -> [LINEAR LAYER]

    Args:
        input_size (int):   Dimension of inputs to be given to the network
        output_size (int):  The dimension of the output
        n_layers (int):     The number of hidden layers of the network
        size (int):         The size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.
    """

    model = nn.Sequential(nn.Linear(in_features=input_size, out_features=size), nn.ReLU())
    for i in range(n_layers-1):
        model.append(nn.Linear(in_features=size, out_features=size))
        model.append(nn.ReLU())
    model.append(nn.Linear(in_features=size, out_features=output_size))
    # model.append(nn.Softmax())

    return model

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)

    Args:
        x (np.ndarray):                 A numpy array to be converted to a torch tensor
        cast_double_to_float (bool):    If True, casts float64 to float32 (default True)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
