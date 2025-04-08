import numpy as np
import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, data, data_ts, seq_len, stride_len, label=None, mode='train'):
        self.mode = mode
        self.data = np.array(data, dtype=np.float32)
        self.data_ts = np.array(data_ts, dtype=np.float32)
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.label = np.array(label, dtype=np.float32) if label is not None else None

    def __len__(self):
        return (self.data.shape[0] - self.seq_len) // self.stride_len + 1

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.stride_len
        end = start + self.seq_len
        
        x = torch.tensor(self.data[start:end])
        x_ts = torch.tensor(self.data_ts[start:end])

        item = {
            'x': x,
            'x_ts': x_ts
        }
        
        if self.mode == 'test':
            item['label'] = torch.tensor(self.label[start:end])
        
        return item