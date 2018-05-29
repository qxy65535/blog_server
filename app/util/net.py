# -*- coding: UTF-8 -*-

import os
from utils.norm import *

root_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
path = os.path.join(os.path.abspath(root_path) + "/util/", "GL121-L1.pkl")
# print(path)
net_gl121 = torch.load(path)

path = os.path.join(os.path.abspath(root_path) + "/util/", "SG-L1.pkl")
net_sg = torch.load(path)

path = os.path.join(os.path.abspath(root_path) + "/util/", "NO-REGU.pkl")
net_noregu = torch.load(path)