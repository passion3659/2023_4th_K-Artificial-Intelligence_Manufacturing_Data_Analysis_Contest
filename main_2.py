#%%
import os
import numpy as np
import pandas as pd

from utils import *

# read data
full_path  = os.getcwd()
data_path  = os.path.join(full_path, 'data', '경진대회용 주조 공정최적화 데이터셋.csv')
data = pd.read_csv(data_path, encoding='cp949')



#%%

# train valid test split
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(data, valid_size=0.2, test_size=0.2, random_state=42)

# preprocess
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

# remove outlier
X_train = remove_outlier(X_train)

# imputation
X_train, X_valid, X_test = imputation(X_train, X_valid, X_test)

