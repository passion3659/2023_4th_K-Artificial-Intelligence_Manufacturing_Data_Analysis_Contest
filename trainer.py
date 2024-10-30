import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss(), label_scaling = False, save_model_path='model_saved/best_model.pth'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)  # 모델을 GPU로 이동
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.label_scaling = label_scaling 
        self.save_model_path = save_model_path

        # 모델을 저장할 디렉토리 확인 및 생성
        save_dir = os.path.dirname(self.save_model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train_step(self, inputs, targets):
        # 모델을 학습 모드로 설정
        self.model.train()

        # 데이터를 GPU로 이동
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # 옵티마이저 초기화
        self.optimizer.zero_grad()

        # 모델의 예측값 계산
        outputs = self.model(inputs)

        # 손실 계산
        loss = self.loss_fn(outputs, targets)

        # 역전파 수행 및 가중치 갱신
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, inputs, targets):
        # 모델을 평가 모드로 설정
        self.model.eval()

        # 데이터를 GPU로 이동
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # 평가 시에는 gradients 계산 필요 없음
        with torch.no_grad():
            # 모델의 예측값 계산
            outputs = self.model(inputs)

            # 손실 계산
            loss = self.loss_fn(outputs, targets)

        return loss.item()

    def fit(self, train_loader, val_loader, epochs=10):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
            for inputs, targets, _, _, _ in train_loop:
                loss = self.train_step(inputs, targets)
                train_loss += loss
                train_loop.set_postfix(train_loss=train_loss / len(train_loader))

            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
            for inputs, targets, _, _, _ in val_loop:
                loss = self.eval_step(inputs, targets)
                val_loss += loss
                val_loop.set_postfix(val_loss=val_loss / len(val_loader))

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 최저 검증 손실 업데이트 및 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f"Best model saved with Val Loss: {best_val_loss:.4f} at {self.save_model_path}")

    def inference(self, data_loader, scaler=None):
        # Check if label_scaling is True but no scaler is provided
        if self.label_scaling and scaler is None:
            raise ValueError("Label scaling is enabled, but no scaler is provided for inverse transformation.")

        self.model.eval()  # 모델을 평가 모드로 설정
        pred_dict = {}
        actual_dict = {}
        
        channel_1_outputs = []
        channel_2_outputs = []
        actual_channel_1_outputs = []
        actual_channel_2_outputs = []

        with torch.no_grad():  # Inference 시 gradient 계산 비활성화
            for i, (X, y, id, time_X, time_y) in enumerate(tqdm(data_loader)):
                
                inputs = X.to(self.device) # b,t,f
                targets = y.to(self.device) # b,t,f
                
                # 모델 예측 수행
                outputs = self.model(inputs) # b,t,f

                # 모델의 출력 채널 수에 따라 분리
                if outputs.shape[2] == 2:  # 채널이 2개인 경우
                    predicted_channel_1 = outputs[:, :, 0].view(-1, 1)  # 채널 1의 예측값
                    predicted_channel_2 = outputs[:, :, 1].view(-1, 1)  # 채널 2의 예측값

                    # 실제 값을 채널별로 분리
                    actual_channel_1 = targets[:, :, 0].view(-1, 1)  # 채널 1의 실제값
                    actual_channel_2 = targets[:, :, 1].view(-1, 1)  # 채널 2의 실제값

                    # 예측값과 실제값을 리스트에 추가
                    channel_1_outputs.append(predicted_channel_1)
                    channel_2_outputs.append(predicted_channel_2)
                    actual_channel_1_outputs.append(actual_channel_1)
                    actual_channel_2_outputs.append(actual_channel_2)
                elif outputs.shape[2] == 1:  # 채널이 1개인 경우
                    if self.label_scaling and scaler:
                        targets = targets.cpu().numpy()
                        batch_size, time_stamp, feature = targets.shape
                        targets = targets.reshape(-1, feature)
                        targets = scaler.inverse_transform(targets)
                        targets = targets.reshape(batch_size, time_stamp, feature)
                        targets = np.transpose(targets, (0, 2, 1))
                        targets = torch.tensor(targets)
                        
                        outputs = outputs.cpu().numpy()
                        batch_size, time_stamp, feature = outputs.shape
                        outputs = outputs.reshape(-1, feature)
                        outputs = scaler.inverse_transform(outputs)
                        outputs = outputs.reshape(batch_size, time_stamp, feature)
                        outputs = np.transpose(outputs, (0, 2, 1))
                        outputs = torch.tensor(outputs)
                    else:
                        targets = targets.permute(0,2,1)
                        outputs = outputs.permute(0,2,1)
                    
                    for batch_idx in range(outputs.size(0)):  
                        file_id = id[batch_idx].item()  
                        time_index = time_y[batch_idx] 
                        
                        if file_id not in pred_dict:
                            pred_dict[file_id] = {}
                        
                        pred_dict[file_id][time_index] = outputs[batch_idx].squeeze(0).cpu().numpy()
                        
                    for batch_idx in range(targets.size(0)):  
                        file_id = id[batch_idx].item() 
                        time_index = time_y[batch_idx]
                        
                        if file_id not in actual_dict:
                            actual_dict[file_id] = {}
                        
                        actual_dict[file_id][time_index] = targets[batch_idx].squeeze(0).cpu().numpy()

        return pred_dict, actual_dict

    def evaluate_metrics(self, val_loader, scaler=None, model_path='model_saved/best_model.pth'):
        # Check if label_scaling is True but no scaler is provided
        if self.label_scaling and scaler is None:
            raise ValueError("Label scaling is enabled, but no scaler is provided for inverse transformation.")

        # 모델 가중치 로드
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)  # 모델을 GPU로 이동
        self.model.eval()  # 평가 모드로 전환

        true_values_channel1 = []
        true_values_channel2 = []
        predictions_channel1 = []
        predictions_channel2 = []
        
        with torch.no_grad():  # 평가 시에는 gradient 계산 불필요
            for inputs, targets, _, _, _ in val_loader:
                # 데이터를 GPU로 이동
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 모델 예측 수행
                outputs = self.model(inputs)

                # 채널 수에 따라 분리
                if outputs.shape[2] == 2:  # 채널이 2개인 경우
                    predicted_channel1 = outputs[:, :, 0].cpu().numpy()
                    predicted_channel2 = outputs[:, :, 1].cpu().numpy()

                    true_channel1 = targets[:, :, 0].cpu().numpy()
                    true_channel2 = targets[:, :, 1].cpu().numpy()

                    true_values_channel1.append(true_channel1)
                    true_values_channel2.append(true_channel2)
                    predictions_channel1.append(predicted_channel1)
                    predictions_channel2.append(predicted_channel2)

                elif outputs.shape[2] == 1:  # 채널이 1개인 경우
                    predicted_channel1 = outputs[:, :, 0].cpu().numpy()
                    true_channel1 = targets[:, :, 0].cpu().numpy()

                    true_values_channel1.append(true_channel1)
                    predictions_channel1.append(predicted_channel1)

        # 리스트를 배열로 변환
        true_values_channel1 = np.concatenate(true_values_channel1, axis=0)
        predictions_channel1 = np.concatenate(predictions_channel1, axis=0)

        # 채널 2가 있는 경우에만 처리
        if true_values_channel2 and predictions_channel2:
            true_values_channel2 = np.concatenate(true_values_channel2, axis=0)
            predictions_channel2 = np.concatenate(predictions_channel2, axis=0)

        # Apply inverse_transform if label_scaling is True and scaler is provided
        if self.label_scaling and scaler:
            print('label_scaler applied for inverse transform')
            predictions_channel1 = scaler.inverse_transform(predictions_channel1)
            true_values_channel1 = scaler.inverse_transform(true_values_channel1)

            if true_values_channel2 and predictions_channel2:
                predictions_channel2 = scaler.inverse_transform(predictions_channel2)
                true_values_channel2 = scaler.inverse_transform(true_values_channel2)

        # 각 채널에 대해 MSE, RMSE, MAE, MAPE 계산
        def calculate_metrics(predictions, true_values):
            mse = np.mean((predictions - true_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - true_values))
            epsilon = 1e-8  # 작은 값을 더해 나눗셈의 안정성 보장
            mape = np.mean(np.abs((predictions - true_values) / (true_values + epsilon))) * 100
            return mse, rmse, mae, mape

        # 채널 1에 대한 결과
        mse1, rmse1, mae1, mape1 = calculate_metrics(predictions_channel1, true_values_channel1)
        print(f"Channel 1\nMSE: {mse1:.4f}\nRMSE: {rmse1:.4f}\nMAE: {mae1:.4f}\nMAPE: {mape1:.4f}%\n")

        # 채널 2가 있는 경우에만 결과 출력
        if true_values_channel2 and predictions_channel2:
            mse2, rmse2, mae2, mape2 = calculate_metrics(predictions_channel2, true_values_channel2)
            print(f"Channel 2\nMSE: {mse2:.4f}\nRMSE: {rmse2:.4f}\nMAE: {mae2:.4f}\nMAPE: {mape2:.4f}%")
