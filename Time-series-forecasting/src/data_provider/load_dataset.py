import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    time_embedding: list = [True, 'h'],
    del_feature: list = None
):
    df = pd.read_csv(os.path.join(datadir, dataname))
    n = len(df)
    trn_idx = int(split_rate[0] * n)
    val_idx = int((split_rate[0] + split_rate[1]) * n)
    split_indices = [trn_idx, val_idx]

    ts = time_features_from_date(df['date'], timeenc=time_embedding[0], freq=time_embedding[1])
    df = df.drop(columns=['date'])

    trn, val, tst = np.split(df, split_indices)
    trn_ts, val_ts, tst_ts = np.split(ts, split_indices)
    var = df.shape[1]

    return trn, trn_ts, val, val_ts, tst, tst_ts, var