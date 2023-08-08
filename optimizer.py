import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml


def get_optimizer(model_params, option_optimizer: dict):
    name = option_optimizer.get('name', None)
    if name is None:
        raise NotImplementedError(
            'Optimizer is None. Please, add to config file')
    name = name.lower()

    optimizer = None
    if name == 'sgd':
        lr = float(option_optimizer.get('lr'))
        momentum = float(option_optimizer.get('momentum', 0.0))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))

        optimizer = torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)

        print('SGD optimizer')

    elif name == 'adam':
        lr = float(option_optimizer.get('lr'))
        beta1 = float(option_optimizer.get('beta1', 0.9))
        beta2 = float(option_optimizer.get('beta2', 0.999))

        Class_optimizer = torch.optim.Adam
        optimizer = Class_optimizer(model_params, lr=lr, betas=(
            beta1, beta2))

        print('Adam optimizer')
    else:
        raise NotImplementedError(
            f'Optimizer [{name}] is not recognized. optimizers.py doesn\'t know {[name]}')

    return optimizer