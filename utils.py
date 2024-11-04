import os
import random
import yaml
import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_config(model_name):
    full_path = os.getcwd()
    config_path = os.path.join(full_path, 'config', 'config.yml')
    print(config_path)
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config[model_name]

def split(df, valid_size=0.2, test_size=0.2, random_state=None):
    # (X: features, y: passorfail)
    X = df.drop(columns=['passorfail'])  
    y = df['passorfail']
    
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    valid_adjusted_size = valid_size / (1 - test_size)  
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_adjusted_size, random_state=random_state)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def preprocess(df) : 
    columns_to_drop = ['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'registration_time', 'tryshot_signal', 'count']
    df = df.drop(columns=columns_to_drop)

    df.rename(columns={'time': 'temp_time', 'date': 'time'}, inplace=True)
    df.rename(columns={'temp_time': 'date'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.drop(columns=['time', 'date'])
    df['working'] = df['working'].apply(lambda x: 0 if x == '가동' else 1 if x == '정지' else x)
    df = pd.get_dummies(df, columns=['heating_furnace'], prefix='heating_furnace', dummy_na=False)
    df[['heating_furnace_A', 'heating_furnace_B']] = df[['heating_furnace_A', 'heating_furnace_B']].astype(int)

    return df

def make_time_series(data, time_threshold=3000):
    """
    Splits the DataFrame into time series based on a time difference threshold.

    Parameters:
    - data: pd.DataFrame containing a 'datetime' column.
    - time_threshold: Time difference in seconds to split the time series (default is 3000 seconds).

    Returns:
    - time_series_dict: A dictionary with DataFrames split by the time threshold.
    """
    # Calculate the time difference in seconds
    data['time_diff'] = data['datetime'].diff().dt.total_seconds()

    time_series_dict = {}
    start_idx = 0

    # Iterate through the DataFrame to find gaps
    for idx in range(1, len(data)):
        if data.loc[idx, 'time_diff'] > time_threshold:  # If the gap is greater than the threshold
            # Extract the interval without 'time_diff' and store it in the dictionary
            time_series_dict[len(time_series_dict)] = data.iloc[start_idx:idx].drop(columns=['time_diff']).reset_index(drop=True)
            start_idx = idx

    # Add the last interval without 'time_diff'
    time_series_dict[len(time_series_dict)] = data.iloc[start_idx:].drop(columns=['time_diff']).reset_index(drop=True)

    # Drop the 'time_diff' column from the original DataFrame
    data.drop(columns=['time_diff'], inplace=True)

    return time_series_dict


def preprocess_time_series(time_series_dict):
    """
    Preprocess the time series dictionary by removing specific indices and merging certain DataFrames.

    Parameters:
    - time_series_dict: A dictionary of DataFrames to preprocess.

    Returns:
    - time_series_dict: The preprocessed dictionary of DataFrames.
    """

    # Remove specified indices
    indices_to_remove = [0, 1, 6, 19, 26, 27, 114, 118, 135, 145, 146]
    for idx in indices_to_remove:
        if idx in time_series_dict:
            del time_series_dict[idx]

    # Merge specified DataFrames
    if 9 in time_series_dict and 10 in time_series_dict:
        time_series_dict[9] = pd.concat([time_series_dict[9], time_series_dict[10]]).reset_index(drop=True)
        del time_series_dict[10]

    if 14 in time_series_dict and 15 in time_series_dict:
        time_series_dict[14] = pd.concat([time_series_dict[14], time_series_dict[15]]).reset_index(drop=True)
        del time_series_dict[15]

    if 137 in time_series_dict and 138 in time_series_dict:
        time_series_dict[137] = pd.concat([time_series_dict[137], time_series_dict[138]]).reset_index(drop=True)
        del time_series_dict[138]
    
    # separate dataframes
    num = len(time_series_dict)
    df = time_series_dict[8]
    df1 = df.iloc[:316]
    df2 = df.iloc[316:]
    time_series_dict[8] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[30]
    df1 = df.iloc[:441]
    df2 = df.iloc[441:]
    time_series_dict[30] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[42]
    df1 = df.iloc[:557]
    df2 = df.iloc[557:]
    time_series_dict[42] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[47]
    df1 = df.iloc[:892]
    df2 = df.iloc[892:]
    time_series_dict[47] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[72]
    df1 = df.iloc[:559]
    df2 = df.iloc[561:]
    time_series_dict[72] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[91]
    df1 = df.iloc[:958]
    df2 = df.iloc[958:]
    time_series_dict[91] = df1
    time_series_dict[num] = df2
    num = num+1

    df = time_series_dict[94]
    df1 = df.iloc[:945]
    df2 = df.iloc[945:]

    time_series_dict[94] = df1
    time_series_dict[num] = df2
    num = num+1
    
    df = time_series_dict[113]
    df1 = df.iloc[:832]
    df2 = df.iloc[832:]
    time_series_dict[113] = df1
    time_series_dict[num] = df2
    num = num+1

    # Sort the dictionary keys to re-index it
    time_series_dict = {i: df for i, (key, df) in enumerate(sorted(time_series_dict.items()), start=0)}

    return time_series_dict

def split_by_process(data_time_series, train_ratio=0.8, val_ratio=0.1):
    """
    Splits data by process into train, validation, and test sets based on the specified ratios.

    Parameters:
    - data_time_series (dict): Dictionary containing time series DataFrames for each process
    - train_ratio (float): Proportion of data to allocate to the train set (default: 0.8)
    - val_ratio (float): Proportion of data to allocate to the validation set (default: 0.1)

    Returns:
    - train_data (dict): Dictionary of DataFrames for each process in the train set
    - val_data (dict): Dictionary of DataFrames for each process in the validation set
    - test_data (dict): Dictionary of DataFrames for each process in the test set
    """
    # Calculate indices for train, validation, and test splits
    num_processes = len(data_time_series)
    train_end = int(num_processes * train_ratio)
    val_end = train_end + int(num_processes * val_ratio)

    # Split data based on calculated indices
    train_data = {key: data_time_series[key] for key in list(data_time_series.keys())[:train_end]}
    val_data = {key: data_time_series[key] for key in list(data_time_series.keys())[train_end:val_end]}
    test_data = {key: data_time_series[key] for key in list(data_time_series.keys())[val_end:]}

    return train_data, val_data, test_data

def interpolate(train_data, val_data, test_data, columns=['molten_temp', 'molten_volume']):
    """
    Fills missing values in specified columns using the median values calculated from the train set.

    Parameters:
    - train_data (dict): Dictionary of DataFrames for each process in the train set
    - val_data (dict): Dictionary of DataFrames for each process in the validation set
    - test_data (dict): Dictionary of DataFrames for each process in the test set
    - columns (list): List of columns to fill missing values (default: ['molten_temp', 'molten_volume'])

    Returns:
    - train_data (dict): Updated train set with missing values filled
    - val_data (dict): Updated validation set with missing values filled
    - test_data (dict): Updated test set with missing values filled
    """
    # Step 1: Concatenate all train DataFrames to calculate median for specified columns
    train_df = pd.concat(train_data.values(), ignore_index=True)
    column_medians = {col: train_df[col].median() for col in columns}
    
    # Step 2: Fill missing values in each set (train, val, test) using the calculated medians
    for data_dict in [train_data, val_data, test_data]:
        for key, df in data_dict.items():
            for col, median_val in column_medians.items():
                df[col] = df[col].fillna(median_val)
    
    return train_data, val_data, test_data


def apply_scaler(train_data, scaler_type='standard'):
    """
    Fits a scaler on the train set, excluding the datetime column, and returns the scaler.
    
    Parameters:
    - train_data (dict): Dictionary of DataFrames for each process in the train set
    - scaler_type (str): Type of scaler to use ('standard' or 'minmax')
    
    Returns:
    - scaler: Fitted scaler object
    """
    # Concatenate all train DataFrames to fit scaler
    train_df = pd.concat(train_data.values(), ignore_index=True)
    
    # Exclude the datetime column from scaling
    feature_columns = train_df.columns.difference(['datetime','passorfail'])
    
    # Choose the scaler type based on the scaler_type argument
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be either 'standard' or 'minmax'")
    
    # Fit the scaler on the feature columns in the train data
    scaler.fit(train_df[feature_columns])
    
    return scaler

def make_dataframe(data, time_interval):
    # List to store selected DataFrames
    selected_dfs = []
    
    for key, df in data.items():
        # Convert the 'datetime' column to datetime format (if necessary)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Select data within the time_interval from the first row's datetime value
        start_time = df['datetime'].iloc[0]
        time_filtered_df = df[df['datetime'] <= start_time + pd.Timedelta(minutes=time_interval)]
        
        # Drop the 'datetime' column
        time_filtered_df = time_filtered_df.drop(columns=['datetime'])
        
        # Append the filtered DataFrame to the list
        selected_dfs.append(time_filtered_df)
    
    # Concatenate all filtered DataFrames into a single DataFrame
    result_df = pd.concat(selected_dfs, ignore_index=True)
    return result_df

def remove_outlier(X, y): 
    condition = (
        (X['upper_mold_temp1'] > 600) |
        (X['upper_mold_temp2'] > 1000) |
        (X['lower_mold_temp2'] > 800) |
        (X['lower_mold_temp3'] > 10000) |
        (X['sleeve_temperature'] > 1200) |
        (X['physical_strength'] > 10000) |
        (X['Coolant_temperature'] > 200)
    )
    
    X_filtered = X[~condition]
    y_filtered = y[~condition]
    
    return X_filtered, y_filtered

def imputation(train, valid, test):
    median_value = train['molten_volume'].median()
    
    train['molten_volume'] = train['molten_volume'].fillna(median_value)
    valid['molten_volume'] = valid['molten_volume'].fillna(median_value)
    test['molten_volume'] = test['molten_volume'].fillna(median_value)
    
    median_value = train['molten_temp'].median()
    train['molten_temp'] = train['molten_temp'].fillna(median_value)
    valid['molten_temp'] = valid['molten_volume'].fillna(median_value)
    test['molten_temp'] = test['molten_temp'].fillna(median_value)
    
    median_value = train['upper_mold_temp3'].median()
    train['upper_mold_temp3'] = train['upper_mold_temp3'].fillna(median_value)
    valid['upper_mold_temp3'] = valid['upper_mold_temp3'].fillna(median_value)
    test['upper_mold_temp3'] = test['upper_mold_temp3'].fillna(median_value)
    
    median_value = train['lower_mold_temp3'].median()
    train['lower_mold_temp3'] = train['lower_mold_temp3'].fillna(median_value)
    valid['lower_mold_temp3'] = valid['lower_mold_temp3'].fillna(median_value)
    test['lower_mold_temp3'] = test['lower_mold_temp3'].fillna(median_value)
    
    return train, valid, test

def save_model(model, model_name):
    # get path
    save_dir = os.path.join(os.getcwd(), "model_saved_ml")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    
    # save
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved at: {model_path}")
    
def load_model(model_name):
    # get path
    model_path = os.path.join(os.getcwd(), "model_saved_ml", f"{model_name}.pkl")
    
    # load
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"No model found at: {model_path}")

