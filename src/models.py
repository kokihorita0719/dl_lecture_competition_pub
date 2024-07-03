import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math



class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

class AdvancedConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size=3, p_drop=0.1),
            ConvBlock(hid_dim, hid_dim * 2, kernel_size=3, p_drop=0.1),
            ConvBlock(hid_dim * 2, hid_dim * 4, kernel_size=3, p_drop=0.1),
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 4, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        encoder_layers = TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=6)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.transformer(X.transpose(1, 2))  # Transformer expects inputs in the shape (S, N, E)
        X = X.transpose(1, 2)  # Transpose back to original shape
        return self.head(X)
    
from torch.nn import LSTM

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.lstm = LSTM(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = X.transpose(1, 2)  # Transpose the last two dimensions
        X, _ = self.lstm(X)
        X = X[:, -1, :]  # Use the last output only
        return self.head(X)

class ConvBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpeechTransformer(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels):
        super().__init__()

        self.num_classes = num_classes  # num_classesをインスタンス変数として設定

        self.conv_blocks = nn.Sequential(
            ConvBlock_2(in_channels, 64),
            ConvBlock_2(64, 128),
            ConvBlock_2(128, 256),
        )

        self.pos_encoder = PositionalEncoding(256, dropout=0.1)

        encoder_layers = TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=6)

        # 全結合層の入力次元はforwardメソッド内で動的に設定する
        self.fc = None

    def forward(self, x, freq_domain_features=None, statistical_features=None):
        x = self.conv_blocks(x)
        x = x.permute(2, 0, 1)  # Transformer expects inputs in the shape (S, N, E)
        x = self.transformer(self.pos_encoder(x))
        x = x.permute(1, 2, 0)  # Transpose back to original shape
        x = x.mean(dim=2)  # Take the mean across the sequence dimension

        # 周波数領域の特徴量と統計的特徴量の次元数を一致させる
        if freq_domain_features is not None and len(freq_domain_features.shape) == 3:
            freq_domain_features = freq_domain_features.mean(dim=2)  # Take the mean across the sequence dimension

        # 周波数領域の特徴量と統計的特徴量を組み込む
        if freq_domain_features is not None and statistical_features is not None:
            x = torch.cat([x, freq_domain_features, statistical_features], dim=1)

        # 全結合層の入力次元を動的に設定
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.num_classes).to(x.device)  # Move self.fc to the same device as x

        return self.fc(x)