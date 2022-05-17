#!/usr/bin/python3

import numpy as np
import torch
import torch.distributed as dist
from typing import List


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    def __init__(self):
        self.consonant_accuracy = AverageMeter()
        self.root_accuracy = AverageMeter()
        self.vowel_accuracy = AverageMeter()
        self.accuracy_avg = AverageMeter()

    def update(self, c_gt, r_gt, v_gt, c_p, r_p, v_p):
        for i in range(len(c_gt)):
            if c_gt[i] == c_p[i]:
                self.consonant_accuracy.update(1)
            else:
                self.consonant_accuracy.update(0)
            if r_gt[i] == r_p[i]:
                self.root_accuracy.update(1)
            else:
                self.root_accuracy.update(0)
            if v_gt[i] == v_p[i]:
                self.vowel_accuracy.update(1)
            else:
                self.vowel_accuracy.update(0)
            if c_gt[i] == c_p[i] and r_gt[i] == r_p[i] and v_gt[i] == v_p[i]:
                self.accuracy_avg.update(1)
            else:
                self.accuracy_avg.update(0)

