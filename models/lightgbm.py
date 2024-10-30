import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def optimize_lightgbm(data, hyperparams):
    # 데이터 로드
    X_train = data['X_train']
    X_valid = data['X_valid']
    y_train = data['y_train']
    y_valid = data['y_valid']

    # LightGBM 모델 설정 (하이퍼파라미터 직접 지정)
    model = lgb.LGBMClassifier(
        num_leaves=hyperparams['num_leaves'],
        learning_rate=hyperparams['learning_rate'],
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        subsample=hyperparams['subsample'],
        colsample_bytree=hyperparams['colsample_bytree'],
        reg_alpha=hyperparams['reg_alpha'],
        reg_lambda=hyperparams['reg_lambda']
    )

    # 모델 학습 (validation 데이터를 사용해 최적화)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    # Validation 데이터로 성능 평가
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
