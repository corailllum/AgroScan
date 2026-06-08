import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PLANTVILLAGE_DIR = os.path.join(BASE_DIR, "PlantVillage_split_equalemment_augmente")
PLANTDOC_DIR = os.path.join(BASE_DIR, "PlantDoc")

MODEL_PATH = os.path.join(BASE_DIR, "models/resnet50_mix_plantdoc.pth")


BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
LR_FINETUNE = 1e-4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


print("\n[1] Loading PlantVillage...")

t0 = time.time()
train_v = datasets.ImageFolder(os.path.join(PLANTVILLAGE_DIR, "train"), transform=train_transform)
val_v   = datasets.ImageFolder(os.path.join(PLANTVILLAGE_DIR, "validation"), transform=val_transform)
test_v  = datasets.ImageFolder(os.path.join(PLANTVILLAGE_DIR, "test"), transform=val_transform)

class_names = train_v.classes
NUM_CLASSES = len(class_names)

print(f"[OK] PlantVillage loaded in {time.time()-t0:.2f}s")
print("Classes:", NUM_CLASSES)


print("\n[2] Loading PlantDoc...")

t0 = time.time()
doc_train = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "train"))
doc_test  = datasets.ImageFolder(os.path.join(PLANTDOC_DIR, "test"))

print(f"[OK] PlantDoc loaded in {time.time()-t0:.2f}s")
print("Train size:", len(doc_train))
print("Test size:", len(doc_test))


print("\n[3] Splitting PlantDoc train/val...")

val_ratio = 0.2
n_val = int(len(doc_train) * val_ratio)
n_train = len(doc_train) - n_val

doc_train, doc_val = torch.utils.data.random_split(
    doc_train,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

print("PlantDoc train:", len(doc_train))
print("PlantDoc val:", len(doc_val))


class SafeDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception as e:
            print("Corrupt image skipped:", e)
            return self.__getitem__((idx + 1) % len(self))

doc_train = SafeDataset(doc_train, train_transform)
doc_val   = SafeDataset(doc_val, val_transform)


print("\n[4] Building loaders...")

train_dataset = ConcatDataset([train_v, doc_train])
val_dataset   = ConcatDataset([val_v, doc_val])
test_dataset  = ConcatDataset([test_v, doc_test])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("[OK] DataLoaders ready")

print("\n[5] Loading model...")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

print("[OK] Model ready")


def run(loader, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if train:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)

            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


print("\n[6] Training start")

best_acc = 0

for epoch in range(5):
    t0 = time.time()

    tr_loss, tr_acc = run(train_loader, True)
    va_loss, va_acc = run(val_loader, False)

    scheduler.step()

    print(f"[P1] Epoch {epoch+1} | "
          f"train {tr_acc:.4f} | val {va_acc:.4f} | "
          f"time {time.time()-t0:.1f}s")

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model")


print("\n[7] Fine tuning layer4 + fc")

for name, p in model.named_parameters():
    if "layer4" in name or "fc" in name:
        p.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FINETUNE
)

#
for epoch in range(5, EPOCHS):
    t0 = time.time()

    tr_loss, tr_acc = run(train_loader, True)
    va_loss, va_acc = run(val_loader, False)

    scheduler.step()

    print(f"[P2] Epoch {epoch+1} | "
          f"train {tr_acc:.4f} | val {va_acc:.4f} | "
          f"time {time.time()-t0:.1f}s")

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model")

