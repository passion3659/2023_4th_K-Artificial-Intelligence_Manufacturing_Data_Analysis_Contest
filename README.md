## Project Title
2024_4th_K-Artificial-Intelligence_Manufacturing_Data_Analysis_Contest

## Introduction
This project analyzes early-stage data to identify factors influencing product quality. Additionally, it develops predictive models for late-stage data to forecast future defects, enhancing quality control and decision-making processes in manufacturing.

## Setup
```shell
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.9.0 seaborn==0.13.2 tqdm==4.66.4 optuna==4.0.0
pip install scikit-learn==1.5.1 xgboost==2.1.2 lightgbm==4.5.0 catboost==1.2.5
```
## Data Preparation
Download the dataset and put it under `data/`. 

## Usage
### path1

To train a model, specify the model name and set the mode to `train`.

```shell
python main_1.py --model lightgbm --mode train
```

To test a trained model (inference), specify the model and set the mode to infer.
```shell
python main_1.py --model lightgbm --mode infer
```
### Available Models
- decisiontree
- randomforest
- xgboost
- lightgbm


Run the code and check the `plots` folder for generated plots.
```shell
python main_1_analysis.py
```

### path2
Adjust the `ratio` and `gamma` values as needed when running the command.

```shell
python main_2.py --ratio 1.0 --gamma 1.0

```
