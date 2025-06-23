import os, json, argparse
import numpy as np
torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import timm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from models.MobileNetV4_Conv_Blur_Medium_Enhanced.utils import Res2TSMBlock, Res2NetBlock, TemporalShift

# Custom imports
from models.MobileNetV4_Conv_Blur_Medium_Enhanced.models import (
    MobileNetV4_Base,
    MobileNetV4_TSM,
    MobileNetV4_Res2Net,
    MobileNetV4_Res2TSM
)
# Import PyTorch baseline models
from models.pytorch_baselines.base_models import (
    MobileNetV2_PyTorch,
    MobileNetV3Small_PyTorch,
    EfficientNetB0_PyTorch,
    EfficientNetB3_PyTorch,
    ResNet50_PyTorch,
    InceptionV3_PyTorch,
    DenseNet121_PyTorch
)

class SpectrogramDataset(Dataset):
    def __init__(self, files, labels, base_dirs, img_size=(224,224)):
        self.files = files; self.labels = labels
        self.base_dirs = base_dirs; self.img_size = img_size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        # load .npy
        for d in self.base_dirs:
            p = os.path.join(d, os.path.basename(fname))
            if os.path.exists(p): arr = np.load(p); break
        arr = np.repeat(arr[...,None],3,axis=-1).astype(np.float32)
        img = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
        img = nn.functional.interpolate(img, size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=[
        'v4_base','v4_tsm','v4_r2n','v4_r2tsm','v4_small','v4_hybrid', # Existing MobileNetV4 variants
        'mnet2_pt', 'mnet3s_pt', 'effb0_pt', 'effb3_pt', 'res50_pt', 'incepv3_pt', 'dnet121_pt' # New PyTorch baselines
    ], required=True)
    p.add_argument('--train-files'); p.add_argument('--train-labels')
    p.add_argument('--val-files');   p.add_argument('--val-labels')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=15)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--data-dir',   default='data/specs/train')
    p.add_argument('--dropout-rate', type=float, default=0.5, help="Dropout rate for model head")
    p.add_argument('--verbose',    action='store_true')
    args = p.parse_args()

    # seeds & device
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # splits
    train_files  = np.load(args.train_files, allow_pickle=True)
    train_labels = np.load(args.train_labels, allow_pickle=True)
    val_files    = np.load(args.val_files,   allow_pickle=True)
    val_labels   = np.load(args.val_labels,   allow_pickle=True)
    base_dirs = [args.data_dir]

    train_loader = DataLoader(SpectrogramDataset(train_files, train_labels, base_dirs),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(SpectrogramDataset(val_files,   val_labels,   base_dirs),
                              batch_size=args.batch_size, shuffle=False)

    # model selection
    params = {'dropout': args.dropout_rate} # For MobileNetV4_Conv_Blur_Medium_Enhanced models
    baseline_params = {'dropout_rate': args.dropout_rate, 'pretrained': True, 'in_chans': 3} # For PyTorch baselines

    if args.model == 'v4_base':
        Model = MobileNetV4_Base
        params['model_key'] = 'mobilenetv4_conv_blur_medium'
    elif args.model == 'v4_tsm':
        Model = MobileNetV4_TSM
        params['model_key'] = 'mobilenetv4_conv_blur_medium'
    elif args.model == 'v4_r2n':
        Model = MobileNetV4_Res2Net
        params['model_key'] = 'mobilenetv4_conv_blur_medium'
    elif args.model == 'v4_r2tsm':
        Model = MobileNetV4_Res2TSM
        params['model_key'] = 'mobilenetv4_conv_blur_medium'
    # New PyTorch Baselines
    elif args.model == 'mnet2_pt':
        Model = MobileNetV2_PyTorch
        params = baseline_params
    elif args.model == 'mnet3s_pt':
        Model = MobileNetV3Small_PyTorch
        params = baseline_params
    elif args.model == 'effb0_pt':
        Model = EfficientNetB0_PyTorch
        params = baseline_params
    elif args.model == 'effb3_pt':
        Model = EfficientNetB3_PyTorch
        params = baseline_params
    elif args.model == 'res50_pt':
        Model = ResNet50_PyTorch
        params = baseline_params
    elif args.model == 'incepv3_pt':
        Model = InceptionV3_PyTorch
        params = baseline_params
    elif args.model == 'dnet121_pt':
        Model = DenseNet121_PyTorch
        params = baseline_params
    # Timm direct models (less configurable head dropout here unless we rebuild them like baselines)
    elif args.model == 'v4_small':
        # For these timm models, dropout is typically part of create_model (e.g. drop_rate)
        # The current setup for these does not use the args.dropout_rate for head.
        # To make it consistent, these would also need to be wrapped in a class like TimmBaselineClassifier
        # or adjust their head manually. For now, keeping original behavior.
        model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=1, in_chans=3, drop_rate=args.dropout_rate)
        # model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features,1), nn.Sigmoid()) # timm already adds sigmoid for num_classes=1 if from_logits=False
        # Forcing a Sigmoid as BCELoss expects probabilities. Timm's num_classes=1 might give logits.
        if model.num_classes == 1 and not hasattr(model.get_classifier(), 'out_act'): # A bit heuristic
             model.classifier = nn.Sequential(model.get_classifier(), nn.Sigmoid())
        model = model.to(device)
    elif args.model == 'v4_hybrid':
        model = timm.create_model('mobilenetv4_hybrid_medium', pretrained=True, num_classes=1, in_chans=3, drop_rate=args.dropout_rate)
        if model.num_classes == 1 and not hasattr(model.get_classifier(), 'out_act'):
             model.classifier = nn.Sequential(model.get_classifier(), nn.Sigmoid())
        model = model.to(device)

    # Instantiate model if not already done (for non-timm direct models)
    if 'Model' in locals() and 'model' not in locals():
        model = Model(**params).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_auc':[], 'val_auc':[]}

    def evaluate(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x,y in loader:
                x = x.to(device)
                p = model(x).cpu().numpy()
                ps.extend(p); ys.extend(y)
        y_true, y_pred = np.array(ys), np.array(ps)>0.5
        return roc_auc_score(ys, ps)

    # training
    for ep in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        tr_auc = evaluate(train_loader)
        va_auc = evaluate(val_loader)
        history['train_auc'].append(tr_auc); history['val_auc'].append(va_auc)
        if args.verbose:
            print(f"Epoch {ep}: train AUC {tr_auc:.3f}, val AUC {va_auc:.3f}")

    # save
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), f"outputs/{args.model}.pth")
    with open(f"outputs/history_{args.model}.json","w") as f:
        json.dump(history, f)
    print("Training complete.")
