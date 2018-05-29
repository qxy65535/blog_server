# -*- coding: UTF-8 -*-

import torch
import torchvision
from torch.autograd import Variable

train_dataset = torchvision.datasets.MNIST(root='../data/mnist/train',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_dataset = torchvision.datasets.MNIST(root='../data/mnist/test',
                                        train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=400,
                                         shuffle=True)  # 将数据打乱
test_data = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=400,
                                         shuffle=True)  # 将数据打乱

print(len(train_data))
k = 0
# for i in train_data:
#     k += 1
#     i[0][0].show()
#     if k == 1:
#         break
for i, (images, labels) in enumerate(train_data):  # 利用enumerate取出一个可迭代对象的内容
    k += 1
    images = Variable(images.view(-1, 28 * 28))
    labels = Variable(labels)
    print(images.size(), labels.size())
    if k == 1:
        break