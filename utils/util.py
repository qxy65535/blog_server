import numpy as np
import torch
from model.net import Net
import torch.nn as nn


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def net_cut(net):
    i = 1
    old_weights = net.get_weights()
    old_bias = net.get_bias()
    weights = []
    bias = []
    input_select = old_weights[0].norm(1, dim=0).gt(0)
    inputs = input_select.nonzero().shape[0]
    # print(inputs)
    net.set_input_select(input_select)
    old_weights[0] = torch.masked_select(old_weights[0], input_select).view(-1, inputs)
    # print(old_weights[0].size())
    for weight in old_weights[1:]:
        # print(weight.size())
        # print(old_weights[i-1].size())
        mask = weight.norm(1,dim=0).gt(0)
        nz = mask.nonzero().shape[0]
        # print(nz)
        weights.append(torch.masked_select(old_weights[i-1], mask.view(-1,1)).view(nz, -1))
        bias.append(torch.masked_select(old_bias[i-1], mask))
        old_weights[i] = torch.masked_select(weight, mask).view(-1, nz)
        i += 1
    weights.append(old_weights[-1])
    bias.append(old_bias[-1])
    # for w in weights:
    #     print(w.size())
    # for b in bias:
    #     print(b.size())
    neurons = [x.norm(1,dim=0).gt(0).nonzero().shape[0]
               for x in net.get_weights()[1:]]
    new_net = Net(inputs, net.classes, neurons, input_select)
    new_net.set_weights(weights)
    new_net.set_bias(bias)
    return new_net


def init_weights(m):
    if type(m) == nn.Linear:
        # print(m)
        nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('leaky_relu'))


def adjust_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.masked_fill_(abs(m.weight.data).le(10**-3),0)

