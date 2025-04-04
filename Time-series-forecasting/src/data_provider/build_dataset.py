from torch.utils.data import Dataset
import numpy as np
import torch

class BuildDataset(Dataset):
    def __init__(self, data, data_ts, seq_len, label_len, pred_len):
        self.data = np.array(data, dtype=np.float32)
        self.data_ts = np.array(data_ts, dtype=np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.valid_window = len(data) - seq_len - pred_len + 1

    def __len__(self):        
        return self.valid_window

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.seq_len
        y_start = idx + self.seq_len - self.label_len
        y_end = idx + self.seq_len + self.pred_len

        seq_x = torch.tensor(self.data[x_start:x_end])
        seq_y = torch.tensor(self.data[y_start:y_end])
        ts_x = torch.tensor(self.data_ts[x_start:x_end])
        ts_y = torch.tensor(self.data_ts[y_start:y_end])

        item = {
            'x': seq_x,
            'y': seq_y,
            'x_ts': ts_x,
            'y_ts': ts_y
        }

        return item
