import os
import yaml
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_config(model_name):
    full_path = os.getcwd()
    config_path = os.path.join(full_path, 'config', 'config.yml')
    with open(config_path, 'r') as file:
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
            # Extract the interval and store it in the dictionary
            time_series_dict[len(time_series_dict)] = data.iloc[start_idx:idx].reset_index(drop=True)
            start_idx = idx

    # Add the last interval
    time_series_dict[len(time_series_dict)] = data.iloc[start_idx:].reset_index(drop=True)

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
    indices_to_remove = [0, 6, 19, 26, 27, 42, 47, 72, 91, 94, 113, 114, 118, 135, 145, 146]
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

    # Sort the dictionary keys to re-index it
    time_series_dict = {i: time_series_dict[k] for i, k in enumerate(sorted(time_series_dict.keys()))}

    return time_series_dict

def make_dataframe(data_time_series):
    """
    Extracts the first 20 rows from each DataFrame in a dictionary and combines them into a single DataFrame.

    Parameters:
    - data_time_series: A dictionary of DataFrames.

    Returns:
    - combined_df: A single DataFrame containing the first 20 rows of each DataFrame in the dictionary.
    """
    # Extract the first 20 rows from each DataFrame and store in a list
    df_list = [df.iloc[:20] for df in data_time_series.values()]
    
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    
    columns_to_drop = ['datetime', 'time_diff']
    combined_df = combined_df.drop(columns=columns_to_drop)
    
    return combined_df


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




