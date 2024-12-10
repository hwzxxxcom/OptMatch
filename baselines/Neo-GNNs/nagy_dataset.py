from ogb.linkproppred import PygLinkPropPredDataset

import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset
import torch
import shutil, os

class NagyDataset(PygLinkPropPredDataset):
    def __init__(self, name, root = 'dataset'):
        if name in ['yeast', 'wordnet', 'dblp', 'youtube']:
            self.name = name
            InMemoryDataset.__init__(name, root)
        else:
            super(NagyDataset, self).__init__(name, root)