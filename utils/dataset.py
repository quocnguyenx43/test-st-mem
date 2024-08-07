import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils import functions as f
from utils import preprocessing as p


class ECGDataset(Dataset):

    def __init__(self,
                 path_list, fs_list, label_list,
                 lead_indices, target_len, target_len_sec, target_fs,
                 preprocessor, device):
        
        self.path_list = path_list
        self.fs_list = fs_list
        self.label_list = label_list
        self.device = device

        self.lead_indices = torch.tensor(lead_indices)
        self.padder = p.PadSequence(target_len_sec=target_len_sec)
        self.resampler = p.Resample(target_len=target_len, target_fs=target_fs)
        self.preprocessor = preprocessor

    def check_dataset(self):
        # fname_not_pkl = [f for f in self.filenames if not f.endswith('.pkl')]
        pass
    
    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int):
        data, _ = f.load_data(self.path_list[idx])
        data = data[self.lead_indices]

        fs = self.fs_list[idx]
        # data = self.padder(data, fs)
        # data = self.resampler(data, fs)
        # data = self.preprocessor(data)

        leads = self.lead_indices
        
        if self.label_list is not None:
            labels = self.label_list[idx]
            return {'input_ecg': data, 'labels': labels, 'lead_indices': leads}
            
        return {'input_ecg': data, 'lead_indices': leads}
    
    
# train, dev, test dataset building
def build_dataset(cfg: dict, split: str) -> ECGDataset:
    path_col = cfg.get('path_col', 'path')
    fs_col = cfg.get('fs_col', 'fs')
    fs_col = cfg.get('fs_col', 'fs')
    label_col = cfg.get('label_col', None)

    index_dir = cfg.get(split + '_csv', './data/index.csv')
    index_df = pd.read_csv(index_dir)

    path_list = index_df[path_col].tolist()
    fs_list = index_df[fs_col].astype(int).tolist()
    label_list = index_df[label_col].tolist() if label_col is not None else None

    leads = cfg.get('leads', [0, 1, 2])
    fs = cfg.get('fs', 250)
    length = cfg.get('len', 2500)
    length_sec = cfg.get('len', 10)

    if split == 'train':
        preprocessor = p.get_data_preprocessor(cfg["train_transforms"])
    else:
        preprocessor = p.get_data_preprocessor(cfg["dev_transforms"])
    preprocessor = p.Compose(preprocessor + [p.ToTensor()])

    return ECGDataset(
        path_list, fs_list, label_list,
        leads, length, length_sec, fs,
        preprocessor
    )


# dataloader building
def get_data_loader(dataset: ECGDataset, mode: str, **kwargs) -> DataLoader:
    len_data = len(dataset)
    batch_size = kwargs['batch_size']
    if batch_size > len_data:
        raise ValueError(f'Not enough data to create 1 batch. batch_size: [{batch_size}], data_size: [{len_data}]')
    is_train = mode == 'train'
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, drop_last=is_train, **kwargs)
