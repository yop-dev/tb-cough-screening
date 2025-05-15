# scripts/train.py
import os, json, argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

# import your dataset & model classes
from dataset import SpectrogramDataset
from models.mobilenetv4 import MobileNetV4_Res2TSM, MobileNetV4_Res2Net, MobileNetV4_TSM
import timm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss  = criterion(preds, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    ys, ps, loss_sum = [], [], 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss_sum += criterion(preds, labels).item() * imgs.size(0)
            ys.extend(labels.cpu().numpy()); ps.extend(preds.cpu().numpy())
    y_true, y_score = np.array(ys), np.array(ps)
    y_pred = (y_score > 0.5).astype(int)
    return {
        "loss":      loss_sum / len(loader.dataset),
        "accuracy":  accuracy_score(y_true, y_pred),
        "auc":       roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred)
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["base","tsm","res2net","res2tsm"], required=True)
    p.add_argument("--train-files"); p.add_argument("--train-labels")
    p.add_argument("--val-files");   p.add_argument("--val-labels")
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--epochs",       type=int, default=15)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) DataLoaders
    train_ds = SpectrogramDataset(
        np.load(args.train_files, allow_pickle=True),
        np.load(args.train_labels, allow_pickle=True),
        base_dirs=["data/specs/train"]
    )
    val_ds   = SpectrogramDataset(
        np.load(args.val_files, allow_pickle=True),
        np.load(args.val_labels, allow_pickle=True),
        base_dirs=["data/specs/train"]
    )
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 2) Model selection
    if args.model == "base":
        key = next(m for m in timm.list_models('*mobilenetv4*') if 'conv_blur_medium' in m)
        model = timm.create_model(key, pretrained=True, num_classes=1, in_chans=3)
        # replace classifier head with sigmoid
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1), nn.Sigmoid())
    elif args.model == "tsm":
        model = MobileNetV4_TSM(key)
    elif args.model == "res2net":
        model = MobileNetV4_Res2Net(key)
    else:
        model = MobileNetV4_Res2TSM(key)

    model.to(device)

    # 3) Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # 4) Training loop
    history = {k: [] for k in ["train_loss","train_auc","val_loss","val_auc"]}
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        tr_metrics = evaluate(model, tr_loader, criterion, device)
        va_metrics = evaluate(model, va_loader, criterion, device)
        print(f"Epoch {ep}: train AUC {tr_metrics['auc']:.3f} | val AUC {va_metrics['auc']:.3f}")
        history["train_loss"].append(tr_loss); history["train_auc"].append(tr_metrics["auc"])
        history["val_loss"].append(va_metrics["loss"]); history["val_auc"].append(va_metrics["auc"])
    
    # 5) Save outputs
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/{args.model}.pth")
    with open(f"outputs/history_{args.model}.json","w") as f:
        json.dump(history, f, indent=2)
