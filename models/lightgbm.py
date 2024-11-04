import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def optimize_lightgbm(data, hyperparams, trials):
    # 데이터 로드
    X_train = data['X_train']
    X_valid = data['X_valid']
    y_train = data['y_train']
    y_valid = data['y_valid']

    def objective(trial):
        # Optuna를 통해 하이퍼파라미터 탐색
        num_leaves = trial.suggest_int("num_leaves", hyperparams['num_leaves']['min'], hyperparams['num_leaves']['max'])
        learning_rate = trial.suggest_float("learning_rate", hyperparams['learning_rate']['min'], hyperparams['learning_rate']['max'])
        n_estimators = trial.suggest_int("n_estimators", hyperparams['n_estimators']['min'], hyperparams['n_estimators']['max'])
        max_depth = trial.suggest_int("max_depth", hyperparams['max_depth']['min'], hyperparams['max_depth']['max'])
        subsample = trial.suggest_float("subsample", hyperparams['subsample']['min'], hyperparams['subsample']['max'])
        colsample_bytree = trial.suggest_float("colsample_bytree", hyperparams['colsample_bytree']['min'], hyperparams['colsample_bytree']['max'])
        reg_alpha = trial.suggest_float("reg_alpha", hyperparams['reg_alpha']['min'], hyperparams['reg_alpha']['max'])
        reg_lambda = trial.suggest_float("reg_lambda", hyperparams['reg_lambda']['min'], hyperparams['reg_lambda']['max'])

        model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        # 모델 학습 (early stopping 추가)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="f1",
            callbacks=[early_stopping(stopping_rounds=10)],
        )

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
    print(best_params)
    model = lgb.LGBMClassifier(
        num_leaves=best_params['num_leaves'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="f1",
        callbacks=[early_stopping(stopping_rounds=10)],
    )

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

def inference_lightgbm(model, data):
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
