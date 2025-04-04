import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(
):
    trn = None
    trn_ts = None
    val = None
    val_ts = None
    test_df = None
    test_ts = None
    var = None
    label = None
    return trn, trn_ts, val, val_ts, test_df, test_ts, var, label

