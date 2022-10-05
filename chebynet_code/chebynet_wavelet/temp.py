# -*- coding:UTF-8 -*-

import pickle as pkl
from pprint import pprint

import numpy as np

with open("data/ind.cora.y", 'rb') as f:
    out = pkl.load(f, encoding='latin1')
    np.set_printoptions(140)
    print(out)
    print(len(list(out)))
