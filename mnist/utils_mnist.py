"""Data loading and training helpers for the MNIST pipeline (same API as utils_cifar10.py).

Inputs are normalised with the MNIST mean and standard deviation, exactly as in the Concrete ML MNIST
example used for the reference model, so that all MNIST models see the same input.
Set the environment variable BLADE_FAKE_DATA=1 to replace MNIST with random tensors (smoke tests).
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

STATS = ((0.1307,), (0.3081,))  # same as the reference (Zama) model
NUM_CLASSES = 10
INPUT_SIZE = 28
IN_CHANNELS = 1

ARC_CONFIGS = {
    # Topology of the "6-layer CNN" of Stoian et al. (2023), Table 1:
    # four 3x3 unpadded convs (8, 8, 16, 16), average pooling after the 2nd and 4th, FC 120, FC 10
    "CNN6-MNIST": {"cfg": [8, 8, "A", 16, 16, "A"], "cls_cfg": [120]},
}

DATA_ROOT = os.environ.get("BLADE_DATA", "./data")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fake_sets(n_train=600, n_test=200):
    """Random tensors with the MNIST shape, only for smoke tests."""
    g = torch.Generator().manual_seed(0)
    mk = lambda n: TensorDataset(torch.rand(n, 1, 28, 28, generator=g) * 2 - 1,
                                 torch.randint(0, 10, (n,), generator=g))
    return mk(n_train), mk(n_train), mk(n_test)


def get_data_loaders(augmentation=True, download=True, batch_size=128, data_root=DATA_ROOT):
    """Train (55k) / validation (5k) split of the MNIST training set, plus the 10k test set."""
    if os.environ.get("BLADE_FAKE_DATA") == "1":
        train_full, val_full, test_set = _fake_sets()
        n_val = 100
    else:
        base_tf = [transforms.ToTensor(), transforms.Normalize(*STATS)]
        train_tf = transforms.Compose(([transforms.RandomCrop(28, padding=2)] if augmentation else []) + base_tf)
        test_tf = transforms.Compose(base_tf)
        train_full = torchvision.datasets.MNIST(root=data_root, train=True, download=download, transform=train_tf)
        val_full = torchvision.datasets.MNIST(root=data_root, train=True, download=False, transform=test_tf)
        test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=download, transform=test_tf)
        n_val = 5000

    # Fixed split, independent of the training seed
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
    train_subset = Subset(train_full, perm[:-n_val])
    val_subset = Subset(val_full, perm[-n_val:])

    kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
    return DataLoader(train_subset, shuffle=True, **kwargs), \
        DataLoader(val_subset, shuffle=False, **kwargs), \
        DataLoader(test_set, shuffle=False, **kwargs)


def _unwrap_out(out):
    return out.tensor if hasattr(out, "tensor") else out


def training_phase(model, train_loader, criterion, optimizer, device, clip_weights=True, post_step=None,
                   teacher=None, kd_temp=2.0, kd_alpha=0.5):
    model.to(device)
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _unwrap_out(model(imgs))
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = _unwrap_out(teacher(imgs))
            kd_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.nn.functional.log_softmax(logits / kd_temp, dim=1),
                torch.nn.functional.softmax(teacher_logits / kd_temp, dim=1)
            ) * (kd_temp * kd_temp)
            loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * criterion(logits, lbls)
        else:
            loss = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if post_step is not None:
            post_step()
        if clip_weights and hasattr(model, "clip_weights"):
            model.clip_weights()
        bs = lbls.size(0)
        running_loss += loss.item() * bs
        total += bs
        correct += (logits.argmax(1) == lbls).sum().item()
    return float(running_loss / total), float(correct / total), float(time.time() - start_time)


def evaluate(model, loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = _unwrap_out(model(x))
            total_loss += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return float(total_loss / total), float(correct / total)


class BestCheckpoint:
    """Saves the best-validation checkpoint; never stops training early."""

    def __init__(self, path="best_model.pth"):
        self.path = path
        self.best_acc = 0.0
        self.early_stop = False  # kept for interface compatibility

    def __call__(self, acc, model):
        if acc > self.best_acc:
            print(f" -> [NEW BEST MODEL] Saved checkpoint to {self.path} (val_acc: {self.best_acc:.4f} -> {acc:.4f})")
            self.best_acc = acc
            torch.save(model.state_dict(), self.path)


def get_calibration_set(val_loader, num_samples=500):
    calib_images, it = [], iter(val_loader)
    while sum(img.shape[0] for img in calib_images) < num_samples:
        try:
            imgs, _ = next(it)
        except StopIteration:
            break
        calib_images.append(imgs.numpy())
    return np.ascontiguousarray(np.concatenate(calib_images, axis=0)[:num_samples], dtype=np.float32)