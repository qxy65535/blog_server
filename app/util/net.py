# -*- coding: UTF-8 -*-

import os
from src.norm import *

root_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
path = os.path.join(os.path.abspath(root_path) + "/util/", "new_net1.pkl")
print(path)
net = torch.load(path)