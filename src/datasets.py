import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torchvision import transforms
from PIL import Image
from scipy.signal import resample, butter, lfilter
from scipy.fftpack import fft


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", sampling_rate: int = 1000) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.sampling_rate = sampling_rate
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        # 必要な引数を設定
        new_fs = 200  # 新しいサンプリングレートを例として200Hzに設定
        lowcut = 10  # フィルタの低周波数カットオフを例として10Hzに設定
        highcut = 100  # フィルタの高周波数カットオフを例として40Hzに設定
        baseline_period = 100  # ベースライン補正の期間を例として100サンプルに設定

        # 前処理されたデータを取得
        preprocessed_data, freq_domain_features, statistical_features = preprocess_eeg_data(
            self.X[i].numpy(), 
            self.sampling_rate, 
            new_fs, 
            lowcut, 
            highcut, 
            baseline_period
        )
        # データがNumPy配列であればテンソルに変換
        if isinstance(preprocessed_data, np.ndarray):
            preprocessed_data = torch.from_numpy(preprocessed_data).float()
        else:
            # データが既にテンソルであればfloat型に変換
            preprocessed_data = preprocessed_data.float()

        # FFTの結果から実部を取り出してテンソルに変換
        freq_domain_features = torch.from_numpy(np.real(freq_domain_features)).float()

        # 統計的特徴量もテンソルに変換
        statistical_features = torch.from_numpy(statistical_features).float()

        if self.split == "train":  # データ拡張は訓練データにのみ適用
            # ノイズの追加
            noise = torch.randn_like(preprocessed_data) * 0.01
            preprocessed_data += noise

            # タイムシフト
            shift = torch.randint(-5, 5, (1,)).item()
            preprocessed_data = torch.roll(preprocessed_data, shifts=shift, dims=0)

        if hasattr(self, "y"):
            return preprocessed_data, freq_domain_features, statistical_features, self.y[i], self.subject_idxs[i]
        else:
            return preprocessed_data, freq_domain_features, statistical_features, self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dir='data'):
        """
        Args:
            split (str): One of 'train' or 'val' indicating the split of dataset.
            data_dir (str): Directory where the image paths text files are stored.
        """
        assert split in ['train', 'val'], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to the same size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
        
        # Load image paths
        with open(os.path.join(data_dir, f'{split}_image_paths.txt'), 'r') as file:
            self.image_paths = file.read().splitlines()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        image = self.transform(image)
        
        return image
    

# バンドパスフィルタ関数を定義
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low < 0 or low > 1 or high < 0 or high > 1:
        raise ValueError("Cutoff frequencies must be in [0, 1], relative to Nyquist frequency.")
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# リサンプリング関数を定義
def resample_data(data, original_fs, new_fs):
    # データが複数のチャネルを持つ場合、各チャネルにリサンプリングを適用
    if data.ndim == 2:
        number_of_samples = round(data.shape[1] * float(new_fs) / original_fs)
        resampled_data = np.array([resample(channel_data, number_of_samples) for channel_data in data])
    else:
        number_of_samples = round(len(data) * float(new_fs) / original_fs)
        resampled_data = resample(data, number_of_samples)
    return resampled_data

# スケーリング関数を定義
def scale_data(data, method='standard'):
    if method == 'standard':
        scaled_data = (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled_data

# ベースライン補正関数を定義
def baseline_correction(data, baseline_period):
    baseline_values = data[:baseline_period]
    corrected_data = data - np.mean(baseline_values)
    return corrected_data

# 全ての前処理ステップを統合する関数を定義
def preprocess_eeg_data(eeg_data, fs, new_fs, lowcut, highcut, baseline_period, scaling_method='standard'):
    # リサンプリング
    resampled_data = resample_data(eeg_data, fs, new_fs)
    
    # バンドパスフィルタリング（カットオフ周波数を相対的な値に変換）
    lowcut_relative = lowcut / (0.5 * new_fs)
    highcut_relative = highcut / (0.5 * new_fs)
    filtered_data = bandpass_filter(resampled_data, lowcut_relative, highcut_relative, new_fs)
    
    # ベースライン補正
    baseline_corrected_data = baseline_correction(filtered_data, baseline_period)
    
    # スケーリング
    scaled_data = scale_data(baseline_corrected_data, method=scaling_method)
    
    # FFTを用いて周波数領域での特徴量を抽出
    freq_domain_features = fft(scaled_data)

    # 統計的特徴量を抽出
    statistical_features = np.array([np.mean(scaled_data), np.std(scaled_data), np.min(scaled_data), np.max(scaled_data)])

    return scaled_data, freq_domain_features, statistical_features