import optuna
from optuna.samplers import TPESampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def optimize_decisiontree(data, hyperparams, trials):
    # 데이터 로드
    X_train = data['X_train']
    X_valid = data['X_valid']
    y_train = data['y_train']
    y_valid = data['y_valid']

    def objective(trial):
        # Optuna를 통해 하이퍼파라미터 탐색
        max_depth = trial.suggest_int("max_depth", hyperparams['max_depth']['min'], hyperparams['max_depth']['max'])
        min_samples_split = trial.suggest_int("min_samples_split", hyperparams['min_samples_split']['min'], hyperparams['min_samples_split']['max'])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", hyperparams['min_samples_leaf']['min'], hyperparams['min_samples_leaf']['max'])
        max_features = trial.suggest_float("max_features", hyperparams['max_features']['min'], hyperparams['max_features']['max'])

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )

        # 모델 학습
        model.fit(X_train, y_train)

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
    model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features']
    )
    model.fit(X_train, y_train)

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

def inference_decisiontree(model, data):
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
