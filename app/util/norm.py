# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable


def l1norm(params):
    l1_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        l1_reg = l1_reg.cuda()
    i = 0
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
    i = 0
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
    i = 0
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
    i = 0
    for param in params:
        # if i % 2 == 0:
        norm1 = param.norm(1, dim=0)
        d = torch.tensor(norm1.nonzero().size()[0]).float()
        # print(norm1.nonzero().size()[0])
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm1.gt(0)

        # if gl121_reg is None:
        #     gl121_reg = torch.sqrt(param.norm(1, dim=0)).sum()
        # else:
        gl121_reg = gl121_reg + torch.sqrt(torch.masked_select(norm1, mask)).sum() + \
                    param.norm(1)
        # params.__next__()
        # i += 1
    # for h in hidden.h:
    #     for param in h.parameters():
    #         gl121_reg = gl121_reg + torch.sqrt(param.norm(1, dim=0)).sum()
    # print(gl121_reg)
    return gl121_reg

def groupl121norm_ori(params):
    gl121_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl121_reg = gl121_reg.cuda()
    i = 0
    for param in params:
        # if i % 2 == 0:
        norm1 = param.norm(1, dim=0)
        d = torch.tensor(param.shape[0]).float()
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm1.gt(0)

        gl121_reg = gl121_reg + torch.sqrt(torch.masked_select(norm1, mask)).sum()
    return gl121_reg


def groupl121norm_l1_ori(params):
    gl121_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl121_reg = gl121_reg.cuda()
    i = 0
    for param in params:
        # if i % 2 == 0:
        norm1 = param.norm(1, dim=0)
        d = torch.tensor(norm1.shape[0] if norm1.shape else 1).float()
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm1.gt(0)

        gl121_reg = gl121_reg + torch.sqrt(torch.masked_select(norm1, mask)).sum() + param.norm(1)
    return gl121_reg


def tmp(params):
    gl121_reg = Variable(torch.tensor(0.), requires_grad=True)
    if torch.cuda.is_available():
        gl121_reg = gl121_reg.cuda()
    i = 0
    for param in params:
        # if i % 2 == 0:
        # weight, bias = param
        mask0 = param.abs().gt(0)
        # print(param[:,1].size())
        # norm1 = torch.sqrt(param.abs()).norm(1, dim=0)
        norm1 = [torch.sqrt(torch.masked_select(param[:, i].abs(), mask0[:, i])).sum() for i in range(param.size()[-1])]


        d = torch.tensor(len(norm1)).float()

        if torch.cuda.is_available():
            d = d.cuda()
        # mask = norm1.gt(0)

        # if gl121_reg is None:
        #     gl121_reg = torch.sqrt(param.norm(1, dim=0)).sum()
        # else:
        for n in norm1:
            if n != 0:
                gl121_reg = gl121_reg + torch.sqrt(d)*torch.sqrt(n)
        # params.__next__()
        # i += 1
    # for h in hidden.h:
    #     for param in h.parameters():
    #         gl121_reg = gl121_reg + torch.sqrt(param.norm(1, dim=0)).sum()
    return gl121_reg

def tmp1(params):
    gl121_reg = Variable(torch.tensor(0.), requires_grad=True)
    delta = torch.tensor(10**-10)
    if torch.cuda.is_available():
        gl121_reg = gl121_reg.cuda()
        delta = delta.cuda()
    i = 0
    for param in params:
        # if i % 2 == 0:
        # mask0 = param.gt(0)
        # param = param
        norm1 = torch.sqrt(param.abs())
        norm1 = norm1.norm(1, dim=0)
        d = torch.tensor(norm1.shape[0] if norm1.shape else 1).float()
        if torch.cuda.is_available():
            d = d.cuda()
        mask = norm1.gt(0)
        #
        # if gl121_reg is None:
        #     gl121_reg = torch.sqrt(param.norm(1, dim=0)).sum()
        # else:
        gl121_reg = gl121_reg + torch.sqrt(d) * torch.sqrt(torch.masked_select(norm1, mask)).sum() + param.norm(1)
        # params.__next__()
        # i += 1
    # for h in hidden.h:
    #     for param in h.parameters():
    #         gl121_reg = gl121_reg + torch.sqrt(param.norm(1, dim=0)).sum()
    return gl121_reg