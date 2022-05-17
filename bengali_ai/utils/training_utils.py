#!/usr/bin/python3

import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def poly_learning_rate2(optimizer, base_lr, curr_iter, max_iter,
                        power=0.9, index_split=-1, scale_lr=10.,
                        warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr

    return optimizer