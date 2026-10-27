import json
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt

import utils_cifar10 as utils

parser = argparse.ArgumentParser(description="PATH B: Train FP32 Teacher")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--save_dir", type=str, default="checkpoints_teacher")
parser.add_argument("--results_dir", type=str, default="results_teacher")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(42)

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.results_dir, exist_ok=True)
base_name = "ResNet18_Teacher_FP32"
model_file_path = os.path.join(args.save_dir, f"{base_name}.pth")

train_loader, val_loader, test_loader = utils.get_data_loaders(augmentation=True, batch_size=args.batch)

# Standard ResNet18 adapted for 32x32 CIFAR-10 images
model = models.resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
stopper = utils.BestCheckpoint(path=model_file_path)

print(f"=== PATH B: FP32 Teacher Training ===")

curves = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": []
}

total_train_start = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    model.train()
    
    correct, total, running_loss = 0, 0, 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, predicted = logits.max(1)
        total += lbls.size(0)
        correct += predicted.eq(lbls).sum().item()
        
    t_loss = running_loss / total
    t_acc = correct / total
    t_time = time.time() - epoch_start
    
    v_start = time.time()
    v_loss, v_acc = utils.evaluate(model, val_loader, criterion, DEVICE)
    v_time = time.time() - v_start
    
    epoch_total_time = time.time() - epoch_start
    current_lr = float(optimizer.param_groups[0]['lr'])
    
    curves["epoch"].append(epoch + 1)
    curves["train_loss"].append(float(t_loss))
    curves["train_acc"].append(float(t_acc))
    curves["val_loss"].append(float(v_loss))
    curves["val_acc"].append(float(v_acc))
    curves["lr"].append(current_lr)
    
    print(f"Ep[{epoch+1:03d}/{args.epochs}] "
          f"t_acc={t_acc:.4f} (train: {t_time:.1f}s) | "
          f"v_acc={v_acc:.4f} (val: {v_time:.1f}s) | "
          f"epoch_total: {epoch_total_time:.1f}s")
    
    scheduler.step()
    stopper(v_acc, model)
    if stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

total_train_time = time.time() - total_train_start

# Final Test Evaluation (Rollback to best checkpoint)
test_start = time.time()
model.load_state_dict(torch.load(model_file_path, map_location=DEVICE))
_, test_acc = utils.evaluate(model, test_loader, criterion, DEVICE)
test_time = time.time() - test_start

print(f"\n[DONE] Teacher Test Acc: {test_acc:.4f} (eval took: {test_time:.2f}s)")

# Save JSON metrics
history = {
    "phase": "B_teacher_fp32",
    "model_tag": base_name,
    "batch_size": args.batch,
    "best_val_accuracy": float(stopper.best_acc),
    "test_accuracy": float(test_acc),
    "total_train_time_sec": round(total_train_time, 2),
    "test_eval_time_sec": round(test_time, 2),
    "epochs_trained": epoch + 1,
    "curves": curves
}

json_path = os.path.join(args.results_dir, f"{base_name}_metrics.json")
with open(json_path, "w") as f:
    json.dump(history, f, indent=4)
print(f"Metrics saved to {json_path}")

# Save Plot PNG
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(curves["epoch"], curves["train_loss"], label="Train Loss")
    ax1.plot(curves["epoch"], curves["val_loss"], label="Val Loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(curves["epoch"], curves["train_acc"], label="Train Acc")
    ax2.plot(curves["epoch"], curves["val_acc"], label="Val Acc", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.results_dir, f"{base_name}_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Curves plot saved to {plot_path}")
except Exception as e:
    print(f"Warning: Plot generation failed: {e}")