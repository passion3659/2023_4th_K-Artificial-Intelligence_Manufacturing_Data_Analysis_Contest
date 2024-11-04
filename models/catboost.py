import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def optimize_catboost(data, hyperparams, trials):
    # 데이터 로드
    X_train = data['X_train']
    X_valid = data['X_valid']
    y_train = data['y_train']
    y_valid = data['y_valid']

    def objective(trial):
        # Optuna를 통해 하이퍼파라미터 탐색
        depth = trial.suggest_int("depth", hyperparams['depth']['min'], hyperparams['depth']['max'])
        learning_rate = trial.suggest_float("learning_rate", hyperparams['learning_rate']['min'], hyperparams['learning_rate']['max'])
        iterations = trial.suggest_int("iterations", hyperparams['iterations']['min'], hyperparams['iterations']['max'])
        subsample = trial.suggest_float("subsample", hyperparams['subsample']['min'], hyperparams['subsample']['max'])
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", hyperparams['min_data_in_leaf']['min'], hyperparams['min_data_in_leaf']['max'])
        leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", hyperparams['leaf_estimation_iterations']['min'], hyperparams['leaf_estimation_iterations']['max'])
        bagging_temperature = trial.suggest_float("bagging_temperature", hyperparams['bagging_temperature']['min'], hyperparams['bagging_temperature']['max'])
        colsample_bylevel = trial.suggest_float("colsample_bylevel", hyperparams['colsample_bylevel']['min'], hyperparams['colsample_bylevel']['max'])

        model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            subsample=subsample,
            min_data_in_leaf=min_data_in_leaf,
            leaf_estimation_iterations=leaf_estimation_iterations,
            bagging_temperature=bagging_temperature,
            colsample_bylevel=colsample_bylevel,
            verbose=0
        )

        # 모델 학습
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)

        # Validation 데이터로 성능 평가
        y_pred_valid = model.predict(X_valid)
        f1 = f1_score(y_valid, y_pred_valid, pos_label=1)

        return f1  # F1 score를 최적화 목표로 설정

    # Optuna 최적화 실행
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    # 최적 파라미터로 모델 학습
    best_params = study.best_params
    model = CatBoostClassifier(
        depth=best_params['depth'],
        learning_rate=best_params['learning_rate'],
        iterations=best_params['iterations'],
        subsample=best_params['subsample'],
        min_data_in_leaf=best_params['min_data_in_leaf'],
        leaf_estimation_iterations=best_params['leaf_estimation_iterations'],
        bagging_temperature=best_params['bagging_temperature'],
        colsample_bylevel=best_params['colsample_bylevel'],
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)

    # Validation 데이터로 최종 성능 평가
    y_pred_valid = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred_valid)
    precision = precision_score(y_valid, y_pred_valid, pos_label=1)
    recall = recall_score(y_valid, y_pred_valid, pos_label=1)
    f1 = f1_score(y_valid, y_pred_valid, pos_label=1)
    conf_matrix = confusion_matrix(y_valid, y_pred_valid)
    classification_rep = classification_report(y_valid, y_pred_valid)

    # Validation 성능 지표를 딕셔너리로 저장
    score = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_rep
    }

    # 최적화된 모델과 validation score 반환
    return model, score

def inference_catboost(model, data):
    # 데이터 로드 (최종 테스트 데이터)
    X_test = data['X_test']
    y_test = data['y_test']

    # 테스트 데이터에 대해 예측
    y_pred = model.predict(X_test)

    # 성능 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # 성능 지표를 딕셔너리로 저장
    score = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_rep
    }

    return score
