# collect_val_errors.py
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.data import PairNPZDataset
from models.simple_compare_cnn import CompareNet

def collect_val_errors():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="./val_error_analysis")
    ap.add_argument("--num_samples", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载验证集
    val_dataset = PairNPZDataset(os.path.join(args.data_dir, "val.npz"), is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompareNet(feat_dim=128).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    misclassified_samples = []
    
    with torch.no_grad():
        for batch_idx, (xa, xb, y) in enumerate(val_loader):
            xa = xa.to(device).float()
            xb = xb.to(device).float()
            y = y.to(device).float()
            
            logits = model(xa, xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            # 找出错误样本
            wrong_mask = (preds != y.long())
            wrong_indices = torch.where(wrong_mask)[0]
            
            for idx in wrong_indices:
                global_idx = batch_idx * val_loader.batch_size + idx.item()
                misclassified_samples.append({
                    'global_index': global_idx,
                    'xa': xa[idx].cpu().numpy(),
                    'xb': xb[idx].cpu().numpy(),
                    'true_label': y[idx].cpu().numpy(),
                    'pred_label': preds[idx].cpu().numpy(),
                    'probability': probs[idx].cpu().numpy()
                })
                
                if len(misclassified_samples) >= args.num_samples:
                    break
            if len(misclassified_samples) >= args.num_samples:
                break
    
    # 可视化
    visualize_val_errors(misclassified_samples, args.save_dir)
    return misclassified_samples

def visualize_val_errors(samples, save_dir):
    n_samples = len(samples)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, sample in enumerate(samples):
        if i >= len(axes):
            break
            
        # 组合图像
        combined_img = np.hstack([sample['xa'][0], sample['xb'][0]])
        
        axes[i].imshow(combined_img, cmap='gray')
        axes[i].set_title(
            f"Idx: {sample['global_index']}\n"
            f"True: {int(sample['true_label'])} | Pred: {int(sample['pred_label'])}\n"
            f"Prob: {sample['probability'][0]:.3f}",
            fontsize=9
        )
        axes[i].axis('off')
        axes[i].axvline(x=27.5, color='red', linestyle='--', alpha=0.7)
    
    # 隐藏多余子图
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_misclassified_samples.png"), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    collect_val_errors()