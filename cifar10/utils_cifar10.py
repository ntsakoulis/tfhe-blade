import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Zama-compatible Scaling
STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
NUM_CLASSES = 10
INPUT_SIZE = 32

ARC_CONFIGS = {
    "VGG9-C10-ZAMA": {
        # Removed the last "A" (Pooling) to allow natural reduction to 1x1
        "cfg": [64, 64, "A", 128, 128, "A", 256, 256], 
        "cls_cfg": [512, 512]
    }
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data_loaders(augmentation=True, download=True, batch_size=128, data_root=os.environ.get("BLADE_DATA", "./data")):
    base_tf = [transforms.ToTensor(), transforms.Normalize(*STATS)]
    if augmentation:
        train_tf_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
        ] + base_tf
    else:
        train_tf_list = base_tf

    train_tf = transforms.Compose(train_tf_list)
    test_tf  = transforms.Compose(base_tf)

    # Two views of the same 50k images: augmented for training, clean for validation
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download, transform=train_tf)
    val_full   = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=test_tf)
    test_set   = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download, transform=test_tf)

    # Fixed split, independent of the training seed
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
    train_subset = Subset(train_full, perm[:45000])
    val_subset   = Subset(val_full, perm[45000:])

    kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())

    return DataLoader(train_subset, shuffle=True, **kwargs), \
           DataLoader(val_subset, shuffle=False, **kwargs), \
           DataLoader(test_set, shuffle=False, **kwargs)

def _unwrap_out(out):
    return out.tensor if hasattr(out, "tensor") else out

def training_phase(model, train_loader, criterion, optimizer, device, clip_weights=True, post_step=None, teacher=None, kd_temp=2.0, kd_alpha=0.5):
    model.to(device)
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()
    
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        logits = _unwrap_out(model(imgs))
        
        # KD Logic: Soften the blow of L1-projection using the original Dense model's probabilities
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = _unwrap_out(teacher(imgs))
            
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.nn.functional.log_softmax(logits / kd_temp, dim=1),
                torch.nn.functional.softmax(teacher_logits / kd_temp, dim=1)
            ) * (kd_temp * kd_temp)
            ce_loss = criterion(logits, lbls)
            loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * ce_loss
        else:
            loss = criterion(logits, lbls)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if post_step is not None:
            post_step()

        if clip_weights and hasattr(model, 'clip_weights'):
            model.clip_weights()

        bs = lbls.size(0)
        running_loss += loss.item() * bs
        _, predicted = torch.max(logits, 1)
        total += bs
        correct += (predicted == lbls).sum().item()
        
    return float(running_loss / total), float(correct / total), float(time.time() - start_time)

def evaluate(model, loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = _unwrap_out(model(x))
            loss = criterion(out, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            correct += (out.argmax(1) == y).sum().item()
            total += bs
    return float(total_loss / total), float(correct / total)

class EarlyStopping:
    def __init__(self, patience=30, path='best_model.pth'):
        self.patience = patience
        self.path = path
        self.best_acc = 0.0
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, acc, model):
        if acc > self.best_acc:
            print(f" -> [NEW BEST MODEL] Saved checkpoint to {self.path} (val_acc: {self.best_acc:.4f} -> {acc:.4f})")
            self.best_acc = acc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience: 
                self.early_stop = True

class BestCheckpoint:
    """Saves the best-val checkpoint; never stops training early."""
    def __init__(self, path='best_model.pth'):
        self.path = path
        self.best_acc = 0.0
        self.early_stop = False  # always False, kept for interface compatibility

    def __call__(self, acc, model):
        if acc > self.best_acc:
            print(f" -> [NEW BEST MODEL] Saved checkpoint to {self.path} (val_acc: {self.best_acc:.4f} -> {acc:.4f})")
            self.best_acc = acc
            torch.save(model.state_dict(), self.path)

def get_calibration_set(val_loader, num_samples=500):
    val_iter = iter(val_loader)
    calib_images = []
    np.random.seed(42)
    while sum(img.shape[0] for img in calib_images) < num_samples:
        imgs, _ = next(val_iter)
        calib_images.append(imgs.numpy())
    return np.ascontiguousarray(np.concatenate(calib_images, axis=0)[:num_samples], dtype=np.float32)