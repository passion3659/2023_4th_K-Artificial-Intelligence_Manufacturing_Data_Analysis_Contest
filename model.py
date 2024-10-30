import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class LSTMModel(nn.Module):
    def __init__(self, input_size=25, hidden_size=64, output_size=100, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 출력 레이어 (hidden_size -> output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 레이어 통과
        out, _ = self.lstm(x)  # out: (batch_size, time_stamp, hidden_size)
        
        # 시퀀스의 마지막 타임스탬프에서 나온 출력값 사용
        out_last = out[:, -1, :]  # 마지막 타임스탬프 값 (batch_size, hidden_size)
        
        # 출력 레이어 적용 (hidden_size -> output_size)
        out = self.fc(out_last)  # out: (batch_size, output_size)
        
        out = out.unsqueeze(-1)
        # out = out.view(out.size(0), 50, -1)
        
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_size=25, hidden_size=64, output_size=100, num_layers=2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 출력 레이어 (hidden_size -> output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU 레이어 통과
        out, _ = self.gru(x)  # out: (batch_size, time_stamp, hidden_size)
        
        # 시퀀스의 마지막 타임스탬프에서 나온 출력값 사용
        out_last = out[:, -1, :]  # 마지막 타임스탬프 값 (batch_size, hidden_size)
        
        # 출력 레이어 적용 (hidden_size -> output_size)
        out = self.fc(out_last)  # out: (batch_size, output_size)
        
        out = out.unsqueeze(-1)
        # out = out.view(out.size(0), 50, -1)
        
        return out

class GCNModel(nn.Module):
    def __init__(self, adj_matrix, input_size, hidden_size, gcn_output_size, mlp_output_size):
        super(GCNModel, self).__init__()
        # GCNConv 레이어 정의
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, gcn_output_size)
        
        # MLP 추가 (Fully Connected Layer)
        # 입력 크기를 (gcn_output_size * num_features)로 설정해야 함
        self.mlp = nn.Sequential(
            nn.Linear(gcn_output_size * 25, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_size)
        )
        
        # 인접행렬(numpy)을 tensor로 변환 후 edge_index로 변환
        self.edge_index = self.adj_to_edge_index(adj_matrix)

    def adj_to_edge_index(self, adj_matrix_tensor):
        # 대각선에 1을 채우는 작업을 torch로 수행
        adj_matrix_tensor.fill_diagonal_(1)

        # 이미 adj_matrix_tensor가 torch tensor이므로 변환 필요 없음
        edge_index = torch.nonzero(adj_matrix_tensor, as_tuple=False).t().contiguous()
        
        return edge_index

    def forward(self, x):
        # x는 (batch, 50, 25)에서 각 배치마다의 시계열 데이터
        batch_size, seq_len, num_features = x.size()  # num_features = 25, seq_len = 50
        
        # GCN에 입력할 그래프 형식으로 변환 (특징: (batch_size * num_features, seq_len))
        x = x.transpose(1, 2).reshape(batch_size * num_features, seq_len)
        
        # GCN 처리
        x = self.gcn1(x, self.edge_index)  # (batch_size * num_features, hidden_size)
        x = torch.relu(x)
        x = self.gcn2(x, self.edge_index)  # (batch_size * num_features, gcn_output_size)
        
        # GCN의 출력을 다시 (batch_size, num_features, gcn_output_size)로 변환
        x = x.view(batch_size, num_features, -1)  # (batch_size, 25, gcn_output_size)
        
        # MLP에 넣기 위해 Flatten (batch_size, 25 * gcn_output_size)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        x = x.unsqueeze(-1) 
        
        return x


if __name__ == "__main__":

    x = torch.rand(4, 50, 25)

    model = LSTMModel(output_size=10)

    y = model(x)

    print(y.shape)
    
    # 예시로 인접행렬 numpy 배열 생성 (Adaptive Lasso로 얻은 결과를 넣으시면 됩니다)
    adj_matrix = np.random.rand(25, 25)  # 예시용 랜덤 인접행렬 (25x25)

    # 모델 초기화 (input_size=50, hidden_size=16, gcn_output_size=10, mlp_output_size=10)
    model = GCNModel(adj_matrix=adj_matrix, input_size=50, hidden_size=16, gcn_output_size=25, mlp_output_size=10)

    # Dummy data로 입력과 target 생성 (batch, 50, 25)
    x = torch.randn(4, 50, 25)  # (batch, 50, 25) 형태의 입력 데이터
    output = model(x)

    # 출력 확인
    print(output.shape)  # (4, 10, 1) 형태로 출력되어야 함