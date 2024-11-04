import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerClassifier, self).__init__()
        
        # Transformer encoder layer with batch_first=True
        encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear layer for classification
        self.classifier = nn.Linear(input_dim, 2)  # Output layer for binary classification

    def forward(self, x, attention_mask):

        # Apply the transformer encoder with attention mask
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        
        # Use the [CLS] token output for classification (first token in the sequence)
        cls_token_output = transformer_output[:, 0, :]  # Shape: (batch_size, input_dim)
        
        # Pass through classifier
        logits = self.classifier(cls_token_output)  # Shape: (batch_size, 2)
        
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross-Entropy Loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 정답 클래스에 대한 확률 계산
        pt = torch.exp(-ce_loss)  # 정답 클래스에 대한 예측 확률

        # Focal Loss 계산
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        # Reduction 옵션 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
