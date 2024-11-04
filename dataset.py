from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data, scaler=None, threshold=5, window_minutes=30, stride_minutes=10, length=None, undersampling=False, ratio=1.0):
        self.data = data
        self.scaler = scaler
        self.threshold = threshold
        self.window_seconds = window_minutes * 60  # Convert minutes to seconds
        self.stride_seconds = stride_minutes * 60  # Convert stride to seconds
        self.processed_data = []
        self.length = length

        for key, df in self.data.items():
            if self.scaler:
                feature_columns = df.columns.difference(['datetime', 'passorfail'])
                df[feature_columns] = self.scaler.transform(df[feature_columns])
            
            # Generate time windows and labels, but skip the first window
            self._create_windows(df, skip_first=True)

        # Apply undersampling if specified
        if undersampling:
            self._apply_undersampling(ratio)

        max_length = max(x.shape[0] for x, _, _ in self.processed_data)
        if self.length is not None and max_length > self.length:
            raise ValueError(
                f"The specified length ({self.length}) is shorter than the longest sequence length ({max_length}). "
                f"Please set `length` to at least {max_length}."
            )

        self._pad_and_add_cls_token()

    def _create_windows(self, df, skip_first=False):
        df = df.sort_values(by="datetime").reset_index(drop=True)
        
        start_idx = 0
        first_window = True
        while start_idx < len(df):
            end_idx = start_idx + 1
            
            # Expand the window to the specified window size
            while end_idx < len(df) and (df.loc[end_idx, "datetime"] - df.loc[start_idx, "datetime"]).total_seconds() < self.window_seconds:
                end_idx += 1
            
            x = df.iloc[start_idx:end_idx].drop(columns=["datetime", "passorfail"]).values
            attention_mask = np.ones(x.shape[0], dtype=int)

            # Label calculation based on next window
            y_start = end_idx
            y_end = y_start
            while y_end < len(df) and (df.loc[y_end, "datetime"] - df.loc[y_start, "datetime"]).total_seconds() < self.window_seconds:
                y_end += 1

            if y_end > y_start:
                y = 1 if df.loc[y_start:y_end, "passorfail"].sum() > self.threshold else 0
                
                if not (first_window and skip_first):
                    self.processed_data.append((x, y, attention_mask))

            first_window = False
            # Move the start index forward by the stride
            start_idx += int(self.stride_seconds / (df["datetime"].diff().dt.total_seconds().mean()))

    def _apply_undersampling(self, ratio):
        # Separate indices by class
        class_0_indices = [i for i, (_, y, _) in enumerate(self.processed_data) if y == 0]
        class_1_indices = [i for i, (_, y, _) in enumerate(self.processed_data) if y == 1]

        # Undersample majority class (class 0) based on the specified ratio
        num_class_1 = len(class_1_indices)
        num_class_0 = int(num_class_1 * ratio)

        if num_class_0 < len(class_0_indices):
            np.random.shuffle(class_0_indices)
            class_0_indices = class_0_indices[:num_class_0]

        # Combine undersampled indices
        undersampled_indices = class_0_indices + class_1_indices
        self.processed_data = [self.processed_data[i] for i in undersampled_indices]

    def _pad_and_add_cls_token(self):
        for i in range(len(self.processed_data)):
            x, y, attention_mask = self.processed_data[i]
            
            if self.length is not None:
                if x.shape[0] < self.length:
                    padding = np.zeros((self.length - x.shape[0], x.shape[1]))
                    x = np.vstack([x, padding])
                    attention_mask = np.hstack([attention_mask, np.zeros(self.length - len(attention_mask), dtype=int)])
                else:
                    x = x[:self.length]
                    attention_mask = attention_mask[:self.length]

            cls_token = np.zeros((1, x.shape[1]))
            x = np.vstack([cls_token, x])

            attention_mask = np.hstack([1, attention_mask])

            self.processed_data[i] = (torch.tensor(x, dtype=torch.float32), 
                                      torch.tensor(y, dtype=torch.long), 
                                      torch.tensor(attention_mask, dtype=torch.long))

    def __getitem__(self, idx):
        x, y, attention_mask = self.processed_data[idx]
        return x, y, attention_mask

    def __len__(self):
        return len(self.processed_data)
