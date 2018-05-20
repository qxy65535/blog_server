# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class Hidden(nn.Module):
    def __init__(self, neurons):
        super(Hidden, self).__init__()
        self.h = list()
        for i in range(len(neurons)-1):
            self.h.append(nn.Linear(neurons[i], neurons[i+1]))

    def forward(self, x):
        for layer in self.h:
            x = F.leaky_relu(layer(x))
        return x

    def get_weights(self):
        return [layer.weight.data for layer in self.h]
