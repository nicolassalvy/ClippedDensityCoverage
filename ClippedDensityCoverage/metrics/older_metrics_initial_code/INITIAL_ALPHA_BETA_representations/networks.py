"""
The whole INITIAL_ALPHA_BETA_representations folder is a copy of the
representations folder from the authors of alpha-Precision and beta-Recall,
taken from their github:
https://github.com/vanderschaarlab/evaluating-generative-models/tree/main/representations

Ahmed Alaa, Boris Van Breugel, Evgeny S Saveliev, and Mihaela van der Schaar.
How faithful is your synthetic data? sample-level metrics for evaluating and
auditing generative models. In International Conference on Machine Learning,
pages 290–306. PMLR, 2022.149

Under MIT license, see INITIAL_ALPHA_BETA_LICENSE.txt
"""

# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# """

#   -----------------------------------------
#   Construction of feature representations
#   -----------------------------------------

#   + build_network:
#     --------------
#             |
#             +--------> feedforward_network:
#             |
#             +--------> recurrent_network:
#             |
#             +--------> MNIST_network:

# """

# TODO: add arguments details


from __future__ import absolute_import, division, print_function

# import numpy as np
# import pandas as pd
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import torch

# from torch.autograd import Variable
# import torch.nn.functional as nnf
# from torch.utils.data import random_split
# from torch.optim import SGD
from torch import nn


# from copy import deepcopy
# import time

torch.manual_seed(1)

# Global variables

ACTIVATION_DICT = {
    "ReLU": torch.nn.ReLU(),
    "Hardtanh": torch.nn.Hardtanh(),
    "ReLU6": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ELU": torch.nn.ELU(),
    "CELU": torch.nn.CELU(),
    "SELU": torch.nn.SELU(),
    "GLU": torch.nn.GLU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Softplus": torch.nn.Softplus(),
}


def build_network(network_name, params):

    if network_name == "feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params):
    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(
        torch.nn.Linear(params["input_dim"], params["num_hidden"], bias=False)
    )
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(
            torch.nn.Linear(
                params["num_hidden"], params["num_hidden"], bias=False
            )
        )
        modules.append(ACTIVATION_DICT[params["activation"]])

    # Output layer

    modules.append(
        torch.nn.Linear(params["num_hidden"], params["rep_dim"], bias=False)
    )

    _architecture = nn.Sequential(*modules)

    return _architecture
