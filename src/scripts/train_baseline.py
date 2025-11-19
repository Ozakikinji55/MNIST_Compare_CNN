# train_baseline.py
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

from tqdm import tqdm

from .utils.seed import set_seed
from .utils.data import PairNPZDataset  
from .models.simple_compare_cnn import CompareNet, count_params

def evaluate(model, loader, device, use_aux=False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.to(device).float()
            xb = xb.to(device).float()
            y  = y.to(device).float()
            
            logit=model(xa, xb)
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).long()
            ys.append(y.long().cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = (y_true == y_pred).mean().item()
    # macro-F1
    f1s = []
    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))
    return acc, f1_macro

def maybe_swap_batch(xa, xb, y):
    # Swap augmentation
    b = xa.size(0)
    mask = torch.rand(b, device=xa.device) < 0.5
    xa2, xb2, y2 = xa.clone(), xb.clone(), y.clone()
    xa2[mask], xb2[mask] = xb[mask], xa[mask]
    if y2.dtype.is_floating_point:
        y2[mask] = 1.0 - y2[mask]
    else:
        y2[mask] = 1 - y2[mask]
    return xa2, xb2, y2

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs/baseline")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--swap_aug", action="store_true")
    ap.add_argument("--pos_weight", type=float, default=1.0, help="BCE pos_weight.")
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor.")
    ap.add_argument("--aux_weight", type=float, default=0.4, help="Weight for auxiliary loss.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompareNet(feat_dim=128, scale=1.0, symmetric_aux=True).to(device)
    n_params = int(count_params(model))

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    train_ds = PairNPZDataset(train_path, is_train=True)   #Ensure the output is float32 and normalized
    val_ds   = PairNPZDataset(val_path, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.1)

    # BCE on difference logit; if there is class imbalance, set pos_weight>1
    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Auxiliary head weight (if enabled）
    if args.label_smoothing > 0:
        def bce_with_smoothing(logits, y):
            
        # Using Smooth Label: y_smooth = y * (1 - smooth) + smooth / 2
            y_smooth = y.float() * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, y_smooth)
        criterion_main = bce_with_smoothing
    else:
        criterion_main = bce

    aux_bce = torch.nn.BCEWithLogitsLoss()
    aux_bce = torch.nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=args.use_amp)

    best = {"acc": 0.0, "f1": 0.0, "epoch": -1}
    patience, bad = 5, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xa, xb, y in pbar:
            xa = xa.to(device).float()
            xb = xb.to(device).float()
            y  = y.to(device).float()

            if args.swap_aug:
                xa, xb, y = maybe_swap_batch(xa, xb, y)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=args.use_amp):
                out = model(xa, xb)
                if isinstance(out, tuple):
                    main_logit, aux_logit = out
                    loss_main = criterion_main(main_logit, y.float()) # 使用 criterion_main
                    loss_aux  = aux_bce(aux_logit, y.float())
                    loss = loss_main + args.aux_weight * loss_aux 
                else:
                    loss = criterion_main(out, y.float())

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()), lr=optim.param_groups[0]["lr"])



        scheduler.step()

        acc, f1 = evaluate(model, val_loader, device, use_aux=True)
        print(f"[Val] epoch={epoch} acc={acc:.4f} f1_macro={f1:.4f} (params={n_params})")

        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
                json.dump({
                    "best_val_acc": acc, "best_val_f1": f1,
                    "best_epoch": epoch, "params": n_params
                }, f, indent=2)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Best @ epoch {best['epoch']}: acc={best['acc']:.4f}, f1_macro={best['f1']:.4f}, params={n_params}")

if __name__ == "__main__":
    main()
