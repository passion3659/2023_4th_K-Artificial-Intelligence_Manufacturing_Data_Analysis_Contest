#%%
import os
import numpy as np
import pandas as pd

from utils import *

from models.randomforest import *
from models.lightgbm import *
from models.xgboost import *
from models.catboost import *

# read data
full_path  = os.getcwd()
data_path  = os.path.join(full_path, 'data', '경진대회용 주조 공정최적화 데이터셋.csv')
data = pd.read_csv(data_path, encoding='cp949') 

# preprocess
data = preprocess(data)
data = make_time_series(data, time_threshold=3000) # 50 minutes
data = preprocess_time_series(data)
data = make_dataframe(data)

# train valid test split
X_train, X_valid, X_test, y_train, y_valid, y_test = split(data, valid_size=0.2, test_size=0.2, random_state=42)

# remove outlier
X_train, y_train = remove_outlier(X_train, y_train)

# imputation
X_train, X_valid, X_test = imputation(X_train, X_valid, X_test)

data = {'X_train': X_train, 'X_valid' : X_valid, 'X_test': X_test, 'y_train': y_train, 'y_valid' : y_valid, 'y_test': y_test }

#%%

# select model and mode
model_name = "xgboost"  # "catboost", "randomforest", "xgboost", "lightgbm"
mode = "inference"      # "train", "inference"

# read config
hyperparams = load_config(model_name)

if mode == "train":
    print('==========training mode==========')
    
    if model_name == "xgboost":
        model, val_score = optimize_xgboost(data, hyperparams)
    elif model_name == "randomforest":
        model, val_score = optimize_randomforest(data, hyperparams)
    elif model_name == "xgboost":
        model, val_score = optimize_xgboost(data, hyperparams)
    elif model_name == "lightgbm":
        model, val_score = optimize_lightgbm(data, hyperparams)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Validation 결과 출력
    print("======Validation Scores======")
    for key, value in val_score.items():    
        print(f"{key} \n {value}")
    
    # 모델 저장
    save_model(model, model_name)

elif mode == "inference":
    print('==========inference mode==========')
    
    if model_name == "catboost":
        model = load_model(model_name)
        test_score = inference_catboost(model, data)
    elif model_name == "randomforest":
        model = load_model(model_name)
        test_score = inference_randomforest(model, data)
    elif model_name == "xgboost":
        model = load_model(model_name)
        test_score = inference_xgboost(model, data)
    elif model_name == "lightgbm":
        model = load_model(model_name)
        test_score = inference_lightgbm(model, data)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Test 결과 출력
    print("======Test Scores======")
    for key, value in test_score.items():    
        print(f"{key} \n {value}")

else: 
    raise ValueError(f"Unsupported mode: {mode}")
    
# %%
