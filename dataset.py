import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BasicDataset(Dataset):
    def __init__(self, chunked_data:list, scaler_Xy, label_scaling=False):
        self.chunked_data = chunked_data
        self.scaler_Xy = scaler_Xy
        self.label_scaling = label_scaling
        
    def __len__(self):
        return len(self.chunked_data)
    
    def __getitem__(self, index):
        (id, X, y) = self.chunked_data[index]
        
        # timesteps
        time_X = X.index.to_numpy()
        time_y = y.index.to_numpy()
        
        # data
        X = self.scaler_Xy[0].transform(X)
        if self.label_scaling:
            y = self.scaler_Xy[1].transform(y)
        else: y = y.to_numpy()
        
        # Convert to torch float tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        time_X = torch.tensor(time_X, dtype=torch.float32)
        time_y = torch.tensor(time_y, dtype=torch.float32)
        
        return X, y, id, time_X, time_y