# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable


def l1norm(params):
    l1_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        l1_reg = l1_reg.cuda()
    # i = 0
    for param in params:
        # if i % 2 == 0:
        l1_reg = l1_reg + param.norm(1)
        # i += 1
        # params.__next__()
    return l1_reg


def l2norm(params):
    l2_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        l2_reg = l2_reg.cuda()
    # i = 0
    for param in params:
        # if i % 2 == 0:
        l2_reg = l2_reg + param.pow(2).sum()
        # i += 1
        # params.__next__()
    return l2_reg


def groupl1norm_l1(params):
    gl1_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl1_reg = gl1_reg.cuda()
    # i = 0
    for param in params:
        # if i % 2 == 0:
        norm2 = param.norm(2, dim=0)
        d = torch.tensor(param.shape[0]).float()
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm2.gt(0)

        gl1_reg = gl1_reg + torch.sqrt(d)*torch.masked_select(norm2, mask).sum() + param.norm(1)
        # i += 1
        # params.__next__()
    return gl1_reg


def groupl1norm(params):
    gl1_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl1_reg = gl1_reg.cuda()
    i = 0
    for param in params:
        # if i % 2 == 0:
        norm2 = param.pow(2).norm(1, dim=0)
        d = torch.tensor(param.shape[0]).float()
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm2.gt(0)

        gl1_reg = gl1_reg + torch.sqrt(d)*torch.sqrt(torch.masked_select(norm2, mask)).sum()# + param.norm(1)
        # i += 1
        # params.__next__()
    return gl1_reg


def groupl121norm_l1(params):
    gl121_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl121_reg = gl121_reg.cuda()
    for param in params:
        norm1 = param.norm(1, dim=0)
        mask = norm1.gt(0)

        # gl121_reg = gl121_reg + torch.sqrt(norm1+1).sum() + param.norm(1)
        gl121_reg = gl121_reg + torch.sqrt(torch.masked_select(norm1, mask)).sum() + param.norm(1)

    return gl121_reg

