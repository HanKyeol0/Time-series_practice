import torch
from data_provider.load_dataset import load_dataset
from data_provider.build_dataset import BuildDataset
from utils.scaling import apply_scaling
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

def create_dataloader(
                datadir: str,
                dataname: str,
                modelname: str,
                scaler: str,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool,
                drop_last: bool,
                seq_len: int,
                stride_len: int,
                target: list,
                val_split_rate: float,
                bank_name: list,
                merge_bank: bool,
                time_embedding: list,
                del_feature: list = None
                ):

    """
    목적: 모든 Argument를 사용하여 아래 load_dataset function, BuildDataset class, apply_scaling function, DataLoader class를 이용하여 dataloader를 생성하고 반환
    조건
    - Data provider 폴더 내의 load_dataset.py, build_dataset.py, utils폴더 내의 scaling.py를 수정하여 구현
    - load_dataset에서 trn은 train data, val은 validation data, tst는 test data, var은 feature 수, label은 test 시 anomaly
    ts는 time stamp의미 
    - 결론적으로 next(iter(trn_dataloader)).shape: (batch_size, seq_len, var)가 되어야함.
    - tst_dataloader는 label이 포함되어야함
    - 최대한 범용적으로 사용할 수 있게끔 코드 작성
    """
    
    trn, trn_ts, val, val_ts, tst, tst_ts, var, label = load_dataset(
        datadir=datadir,
        dataname=dataname,
        val_split_rate = val_split_rate,
        time_embedding=time_embedding,
                            )
    
    # scaling (minmax, minmax square, minmax m1p1, standard)
    trn, val, tst = apply_scaling(
        scaler=scaler,
        trn=trn,
        val=val,
        tst=tst
                    )

    # build dataset
    trn_dataset = BuildDataset(trn, trn_ts, seq_len, stride_len)
    val_dataset = BuildDataset(val, val_ts, seq_len, stride_len)
    tst_dataset = BuildDataset(tst, tst_ts, seq_len, stride_len)

    # torch dataloader
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    tst_dataloader = DataLoader(
        tst_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return trn_dataloader, val_dataloader, tst_dataloader, var, label
