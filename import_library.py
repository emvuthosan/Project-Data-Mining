#Run this only when using colab/kaggle (cuz the library version of torch in those different)
# import torch as _t
# import os
# _TORCH = _t.__version__.split('+')[0]
# _CUDA  = ('cu' + _t.version.cuda.replace('.', '')
#           if _t.cuda.is_available() else 'cpu')
# _WHL   = f'https://data.pyg.org/whl/torch-{_TORCH}+{_CUDA}.html'
# os.system('pip install -q torch_geometric')
# os.system(f'pip install -q pyg_lib torch_sparse torch_scatter -f {_WHL}')

import os
import glob
import gc
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, fbeta_score, recall_score,
                             classification_report, confusion_matrix)