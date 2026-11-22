"""Train independent ProSkin specialists on the original HAM10000 lesion split."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "my-models" / "specialists"
PLOTS = ROOT / "graphics" / "proskin-specialists"
ALL_CLASSES = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]
SPECS = {
    "ProSkinA": {"excluded": ["df"], "focus": "nv", "seed": 101},
    "ProSkinB": {"excluded": ["df", "vasc"], "focus": "mel", "seed": 102},
    "ProSkinC": {"excluded": ["df", "vasc"], "focus": "bkl", "seed": 103},
    "ProSkinD": {"excluded": ["mel", "vasc"], "focus": "bcc", "seed": 104},
    "ProSkinE": {"excluded": ["vasc"], "focus": "akiec", "seed": 105},
    "ProSkinF": {"included": ["df", "nv", "bcc", "bkl"], "focus": "df", "seed": 106},
}


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class ProSkinCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 64, 2),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SkinDataset(Dataset):
    def __init__(self, frame, image_paths, classes, transform):
        self.paths = [image_paths[i] for i in frame.image_id]
        lookup = {name: index for index, name in enumerate(classes)}
        self.labels = [lookup[name] for name in frame.dx]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        return self.transform(image), self.labels[index]


def draw_architecture(name, classes):
    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.axis("off")
    stages = [
        "RGB\n3 × 224 × 224", "Conv 3×3 + BN\n32 × 224²",
        "ResBlock ↓2\n64 × 112²", "ResBlock ↓2\n128 × 56²",
        "ResBlock ↓2\n256 × 28²", "ResBlock ↓2\n512 × 14²",
        "GAP + Dropout\n512", f"Linear\n{len(classes)} класса",
    ]
    xs = np.linspace(0.08, 0.92, len(stages))
    for i, (x, label) in enumerate(zip(xs, stages)):
        ax.text(x, 0.55, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.6", fc="#e8f0f8", ec="#4276a4"))
        if i:
            ax.annotate("", (x - 0.065, 0.55), (xs[i-1] + 0.065, 0.55),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title(f"{name}: архитектура ProSkinCNN; выход: {', '.join(classes)}", fontsize=12)
    fig.savefig(PLOTS / f"{name}_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_history(name, history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Обучение")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Проверка")
    axes[0].set_ylabel("Взвешенная ошибка")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], color="#e28e32")
    axes[1].set_ylabel("Macro F1 на проверке")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{name}: обучение")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}_training.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=list(SPECS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA / "HAM10000_metadata.csv")
    assert metadata.dx.unique().tolist() == ALL_CLASSES
    lesions = metadata.groupby("lesion_id").dx.first().reset_index()
    lesions = lesions.sample(frac=1, random_state=42).reset_index(drop=True)
    cutoff = int(len(lesions) * 0.8)
    train_ids = set(lesions.iloc[:cutoff].lesion_id)
    val_ids = set(lesions.iloc[cutoff:].lesion_id)
    assert train_ids.isdisjoint(val_ids)
    image_paths = {p.stem: p for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2")
                   for p in (DATA / folder).glob("*.jpg")}
    missing = set(metadata.image_id) - image_paths.keys()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images")
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}, models={args.models}", flush=True)
    for name in args.models:
        spec = SPECS[name]
        classes = spec.get("included", [c for c in ALL_CLASSES if c not in spec.get("excluded", [])])
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        train_frame = metadata[metadata.lesion_id.isin(train_ids) & metadata.dx.isin(classes)]
        val_frame = metadata[metadata.lesion_id.isin(val_ids) & metadata.dx.isin(classes)]
        train_ds = SkinDataset(train_frame, image_paths, classes, train_tf)
        val_ds = SkinDataset(val_frame, image_paths, classes, val_tf)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        counts = train_frame.dx.value_counts()
        weights = torch.tensor([len(train_frame)/(len(classes)*counts[c]) for c in classes],
                               dtype=torch.float32, device=device)
        model = ProSkinCNN(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history = []
        best_f1, stale = -1, 0
        config = dict(name=name, classes=classes, excluded=spec.get("excluded", []),
                      focus=spec["focus"], seed=spec["seed"], epochs_max=args.epochs,
                      patience=args.patience, train_images=len(train_frame), val_images=len(val_frame),
                      train_class_counts={c: int(counts[c]) for c in classes},
                      architecture="ProSkinCNN, four residual blocks 64/128/256/512",
                      split="original images, lesion_id 80/20, random_state=42",
                      optimizer="AdamW lr=0.001 weight_decay=0.0001",
                      scheduler="CosineAnnealingLR", loss="weighted CrossEntropyLoss")
        (OUT / f"{name}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        draw_architecture(name, classes)
        print(f"START {name}: classes={classes} train={len(train_ds)} val={len(val_ds)}", flush=True)
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            model.train()
            train_loss, train_count = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
                train_count += len(labels)
            model.eval()
            val_loss, val_count, truth, predicted = 0.0, 0, [], []
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    val_count += len(labels)
                    truth.extend(labels.cpu().tolist())
                    predicted.extend(logits.argmax(1).cpu().tolist())
            macro_f1 = f1_score(truth, predicted, labels=list(range(len(classes))),
                                average="macro", zero_division=0)
            history.append(dict(epoch=epoch, train_loss=train_loss/train_count,
                                val_loss=val_loss/val_count, val_macro_f1=macro_f1,
                                lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-started))
            if macro_f1 > best_f1 + 1e-5:
                best_f1, stale = macro_f1, 0
                torch.save(model.state_dict(), OUT / f"{name}_best.pth")
                config["best_epoch"] = epoch
                config["best_val_macro_f1"] = best_f1
            else:
                stale += 1
            scheduler.step()
            with (OUT / f"{name}_history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            draw_history(name, history)
            (OUT / f"{name}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
            print(f"{name} epoch {epoch:02}/{args.epochs}: train={history[-1]['train_loss']:.4f} "
                  f"val={history[-1]['val_loss']:.4f} F1={macro_f1:.4f} "
                  f"best={best_f1:.4f} time={history[-1]['seconds']:.1f}s", flush=True)
            if stale >= args.patience:
                print(f"{name}: early stop after {epoch} epochs", flush=True)
                break
        torch.save(model.state_dict(), OUT / f"{name}_last.pth")
        config["epochs_run"] = len(history)
        (OUT / f"{name}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"DONE {name}: best epoch={config['best_epoch']} F1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
