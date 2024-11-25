import torch
from torch import nn
import math
import numpy as np
import os
import pickle
from options import Options

grid_gap=0.25


cur_args = Options().parse()
minmax_base = cur_args.minmax_base
room_name = cur_args.apt

with open(os.path.join(minmax_base, room_name + "_minmax"), "rb") as min_max_loader:
    min_maxes = pickle.load(min_max_loader)
    min_pos = min_maxes[0][[0, 2]]
    max_pos = min_maxes[1][[0, 2]]

grid_coors_x = np.arange(min_pos[0], max_pos[0], grid_gap)
grid_coors_y = np.arange(min_pos[1], max_pos[1], grid_gap)

