from __future__ import annotations
import os, time, json, argparse, random, math
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
import matplotlib.pyplot as plt

from model import build_model, freeze_backbone, count_trainable_params

# ===== Normalization (ImageNet) =====
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEFAULT_CIFAR10 = ["automobile", "truck", "ship", "cat", "dog"]

# ===== Utils =====
def set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),

            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def load_data_cifar10(selected: List[str], max_total: int,
                      train_tf: transforms.Compose,
                      test_tf: transforms.Compose,
                      root: str = "./data") -> Tuple[Dataset, Dataset, Dataset, List[str]]:
  
    full_tr = CIFAR10(root=root, train=True, download=True)
    full_te = CIFAR10(root=root, train=False, download=True)

    all_classes = full_tr.classes  # ['airplane', 'automobile', ...]
    name2id = {n: i for i, n in enumerate(all_classes)}

    wanted_ids = [name2id[n] for n in selected if n in name2id]
    class_names = [all_classes[i] for i in wanted_ids]
    if len(wanted_ids) == 0:
        raise ValueError("No valid CIFAR-10 classes selected.")

    pools = {cid: [] for cid in wanted_ids}
    for idx, y in enumerate(full_tr.targets):
        if y in pools:
            pools[y].append(idx)

    per = max_total // len(wanted_ids)
    extra = max_total - per * len(wanted_ids)
    chosen = []
    for i, cid in enumerate(wanted_ids):
        chosen += pools[cid][: per + (i < extra)]
    random.shuffle(chosen)

    # split 80/10/10
    n = len(chosen)
    n_tr = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    idx_tr, idx_val, idx_te = chosen[:n_tr], chosen[n_tr:n_tr + n_val], chosen[n_tr + n_val:]

    old2new = {old: i for i, old in enumerate(wanted_ids)}

    class SubsetWithTransform(Dataset):
        def __init__(self, base_ds, indices, tfm):
            self.base, self.indices, self.tfm = base_ds, indices, tfm
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            x, y = self.base.data[self.indices[i]], self.base.targets[self.indices[i]]
            x = transforms.ToPILImage()(x)
            x = self.tfm(x)
            y = old2new[y]
            return x, y

    ds_tr = SubsetWithTransform(full_tr, idx_tr, train_tf)
    ds_val = SubsetWithTransform(full_tr, idx_val, test_tf)
    ds_te  = SubsetWithTransform(full_tr, idx_te,  test_tf)
    return ds_tr, ds_val, ds_te, class_names

# ====== ImageFolder loaders (לדאטה גדול כמו iNaturalist לאחר הוצאה) ======
def build_dataloaders_imagefolder(root: str, img_size: int, batch_size: int, num_workers: int):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(f"expected struct  ImageFolder with train/ and-val/ under: {root}")

    ds_train = ImageFolder(train_dir, transform=make_transforms(img_size, True))
    ds_val   = ImageFolder(val_dir,   transform=make_transforms(img_size, False))

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2
    )
    return ds_train, ds_val, train_loader, val_loader

# ====== MixUp/CutMix ======
def mixup_data(x, y, alpha: float):
    if alpha <= 0.0:
        return x, y, 1.0, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, lam, y_b

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ====== Train/Eval helpers ======
def build_loss(label_smoothing: float):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def build_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay to 0.1 * lr
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 5):
    with torch.no_grad():
        _, pred = logits.topk(k, 1, True, True)  # [B, k]
        correct = pred.eq(targets.view(-1, 1)).any(dim=1).float().sum().item()
        return correct

def plot_curves(train_losses, val_losses, train_acc1, val_acc1, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()

    # Accuracy (Top-1)
    plt.figure()
    plt.plot(train_acc1, label="train@1")
    plt.plot(val_acc1,   label="val@1")
    plt.title("Top-1 Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Acc@1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png")
    plt.close()

# ====== Main ======
def main() -> None:
    """CLI entry-point: trains on either CIFAR-10 tiny or ImageFolder (iNat)"""
    ap = argparse.ArgumentParser()

    ap.add_argument("--classes", nargs="*", default=DEFAULT_CIFAR10,
                    help="Subset of CIFAR-10 classes to use (tiny run).")
    ap.add_argument("--max_total", type=int, default=100,
                    help="Total number of images across all classes (tiny run).")

    ap.add_argument("--dataset_root", type=str, required=False,
                    help="שורש הדאטה בפורמט ImageFolder עם תיקיות train/ ו-val/")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224, help="224/288/300 לאימון גדול")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--freeze_backbone", action="store_true", help="להקפיא backbone בתחילת אימון קטן")

    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0, help="alpha של mixup (למשל 0.2)")
    ap.add_argument("--cutmix", type=float, default=0.0, help="alpha של cutmix (אלטרנטיבי ל-mixup)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="Mixed Precision (מומלץ על GPU)")

    args = ap.parse_args()

    
    set_seed(42)
    device = get_device()
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    #load all the data
    if args.dataset_root:  
        ds_tr, ds_val, dl_tr, dl_val = build_dataloaders_imagefolder(
            args.dataset_root, args.img_size, args.batch_size, args.num_workers
        )
        class_names = ds_tr.classes
        ds_te = None
    else:  
        train_tf = make_transforms(args.img_size, True)
        test_tf  = make_transforms(args.img_size, False)
        ds_tr, ds_val, ds_te, class_names = load_data_cifar10(
            args.classes, args.max_total, train_tf, test_tf, "./data"
        )
        g = torch.Generator(); g.manual_seed(42)
        common_dl = dict(num_workers=0, pin_memory=(device.type == "cuda"), generator=g)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  **common_dl)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **common_dl)

    num_classes = len(class_names)

    model = build_model(num_classes, dropout=0.3, pretrained=True).to(device)
    if args.freeze_backbone:
        freeze_backbone(model, True)
    print("Trainable params:", count_trainable_params(model))

    criterion = build_loss(args.label_smoothing)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, len(dl_tr), warmup_epochs=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ckpt_dir = Path("training/checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots"); plots_dir.mkdir(parents=True, exist_ok=True)

    (ckpt_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    best_top1 = -1.0
    history_tr_loss, history_va_loss = [], []
    history_tr_acc1, history_va_acc1 = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        total_tr = 0

        for images, targets in dl_tr:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            use_mix = (args.mixup > 0.0) or (args.cutmix > 0.0)
            alpha = args.mixup if args.mixup > 0.0 else (args.cutmix if args.cutmix > 0.0 else 0.0)
            if use_mix and alpha > 0.0:
                images, y_a, lam, y_b = mixup_data(images, targets, alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                if use_mix and alpha > 0.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                total_tr += images.size(0)
                if use_mix and alpha > 0.0:
                    pred1 = logits.argmax(dim=1)
                    running_top1 += (pred1 == y_a).float().sum().item()
                else:
                    pred1 = logits.argmax(dim=1)
                    running_top1 += (pred1 == targets).float().sum().item()

            running_loss += loss.item() * images.size(0)

        tr_loss = running_loss / max(1, total_tr)
        tr_acc1 = (running_top1 / max(1, total_tr)) * 100.0
        history_tr_loss.append(tr_loss)
        history_tr_acc1.append(tr_acc1)

        model.eval()
        va_loss_sum = 0.0
        va_total = 0
        va_correct1 = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            for images, targets in dl_val:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)

                va_loss_sum += loss.item() * images.size(0)
                va_total += images.size(0)
                va_correct1 += (logits.argmax(dim=1) == targets).float().sum().item()

        va_loss = va_loss_sum / max(1, va_total)
        va_acc1 = (va_correct1 / max(1, va_total)) * 100.0
        history_va_loss.append(va_loss)
        history_va_acc1.append(va_acc1)

        print(f"[{epoch+1:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"train@1={tr_acc1:.2f}% val@1={va_acc1:.2f}%")

        if va_acc1 > best_top1:
            best_top1 = va_acc1
            torch.save(model.state_dict(), ckpt_dir / "best_efficientnet_v2_s.pt")

    plot_curves(history_tr_loss, history_va_loss, history_tr_acc1, history_va_acc1, plots_dir)

    if not args.dataset_root and 'ds_te' in locals() and ds_te is not None:
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=(device.type == "cuda"))
        model.eval()
        te_total, te_correct1 = 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            for images, targets in dl_te:
                images = images.to(device); targets = targets.to(device)
                logits = model(images)
                te_total += images.size(0)
                te_correct1 += (logits.argmax(dim=1) == targets).float().sum().item()
        te_acc1 = te_correct1 / max(1, te_total) * 100.0
        print(f"[TEST tiny] top-1 = {te_acc1:.2f}%")

if __name__ == "__main__":
    main()
