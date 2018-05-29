# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        # print(m)
        nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('leaky_relu'))


def adjust_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.masked_fill_(abs(m.weight.data).le(10**-3),0)
        # m.bias.data.masked_fill_(abs(m.bias.data/m.weight.data.shape[1]).le(10 ** -3), 0)

def adjust_params(m):
    if type(m) == nn.Linear:
        m.weight.data.masked_fill_(abs(m.weight.data).le(10**-3),0)
        # print(m.weight.data.shape[1])
        m.bias.data.masked_fill_(abs(m.bias.data/m.weight.data.shape[1]).le(10 ** -3), 0)

class Net(nn.Module):
    def __init__(self, inputs, classes, neurons, input_select=None):
        super(Net, self).__init__()
        self.input = nn.Linear(inputs, neurons[0])
        for i in range(len(neurons)-1):
            self.add_module("hidden%d"%i, nn.Linear(neurons[i], neurons[i+1]))
        self.predict = nn.Linear(neurons[-1], classes)
        self.input_select = input_select
        self.inputs = inputs
        self.classes = classes

    def forward(self, x):
        for name, layer in self._modules.items():
            if name != "predict":
                x = F.leaky_relu(layer(x))
        x = self.predict(x)
        return x

    def get_weights(self):
        weights = [layer.weight.data for layer in self._modules.values()]
        return weights

    def get_weights1(self):
        weights = [layer.weight for layer in self._modules.values()]
        return weights

    def get_bias(self):
        bias = [layer.bias.data for layer in self._modules.values()]
        return bias

    def set_weights(self, weights):
        i = 0
        for layer in self._modules.values():
            layer.weight.data = weights[i]
            i += 1

    def set_bias(self, bias):
        i = 0
        for layer in self._modules.values():
            layer.bias.data = bias[i]
            i += 1

    def set_input_select(self, input_select):
        self.input_select = input_select

    def get_input_select(self):
        return self.input_select
