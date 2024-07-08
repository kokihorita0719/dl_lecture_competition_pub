import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Import t-SNE
from sklearn.decomposition import PCA

from src.datasets import ThingsMEGDataset
from src.models import WaveformClassifier, BasicConvClassifier, ConvTransformer
from src.utils import set_seed

N_FEATURES = 60

def to_one_hot(indices, num_classes):
    one_hot = torch.zeros(indices.shape[0], num_classes)
    one_hot.scatter_(1, indices.unsqueeze(1), 1)
    return one_hot

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    # 最初のサンプルを取得
    X, y, subject_idxs = train_set[0]
    print("X.size: ", X.size())

    # データの可視化
    plt.figure(figsize=(10, 4))
    # 最初の3つのチャネルのデータをプロット
    for i in range(10):
        plt.plot(X[i], label=f'Channel {i+1}')
    
    plt.title(f"Class: {y}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend() 
    # plt.show()
    plt.savefig("plot.png")  # プロットを画像ファイルとして保存

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    # model = BasicConvClassifier(
    #     train_set.num_classes, train_set.seq_len, N_FEATURES
    # ).to(args.device)
    # model = WaveformClassifier(
    #     N_FEATURES + 4, args.hidden_size, args.num_layers, train_set.num_classes
    # ).to(args.device)
    model = ConvTransformer(
        nclasses=train_set.num_classes,  # 分類するクラスの数
        ninp=N_FEATURES + 4,  # 入力の次元数
        nhead=8,  # Transformerのヘッドの数
        nhid=1024,  # Transformerの隠れ層の次元数
        nlayers=6,  # Transformerの層の数
    ).to(args.device)  # モデルをデバイス（CPUまたはGPU）に移動

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    # 以下のコードを追加します
    pca = PCA(n_components=N_FEATURES, svd_solver='randomized')  # PCAオブジェクトを作成し、主成分の数を32に設定
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            subject_idxs = to_one_hot(subject_idxs, 4).to(args.device)  # Assuming there are 4 subjects

            # PCAを適用して特徴量の次元を削減
            original_shape = X.shape
            X = X.view(original_shape[0], -1)  # バッチサイズとそれ以外の次元を結合
            X = X.cpu().numpy()  # Convert the tensor to numpy array on CPU
            X = pca.fit_transform(X)
            X = torch.from_numpy(X).to(args.device).view(original_shape[0], N_FEATURES, -1)  # Convert the numpy array back to tensor and reshape it

            subject_idxs = subject_idxs.unsqueeze(2).expand(-1, -1, X.shape[2])  # Expand the dimensions of subject indices to match the dimensions of X
            X = torch.cat([X, subject_idxs], dim=1)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            subject_idxs = to_one_hot(subject_idxs, 4).to(args.device)  # Assuming there are 4 subjects

            # PCAを適用して特徴量の次元を削減
            original_shape = X.shape
            X = X.view(original_shape[0], -1)  # バッチサイズとそれ以外の次元を結合
            X = X.cpu().numpy()  # Convert the tensor to numpy array on CPU
            X = pca.transform(X)
            X = torch.from_numpy(X).to(args.device).view(original_shape[0], N_FEATURES, -1)  # Convert the numpy array back to tensor and reshape it

            subject_idxs = subject_idxs.unsqueeze(2).expand(-1, -1, X.shape[2])  # Expand the dimensions of subject indices to match the dimensions of X
            X = torch.cat([X, subject_idxs], dim=1)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
