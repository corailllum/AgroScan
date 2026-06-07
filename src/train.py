import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#definition des paramètres
DATA_DIR    = "../PlantVillage_split_equalemment_augmente"
MODEL_PATH  = "../models/resnet50_plantvillage_augm.pth"
NUM_CLASSES = 39
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-3
LR_FINETUNE = 1e-4

#choix du device mps ou cuda ou cpu
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print(f"Appareil utilisé : {DEVICE}")

#transformation avec reshape, tansor etc
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#chargement dataset et loader
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),      transform=train_transform)
NUM_CLASSES = len(train_dataset.classes)  # redéfini dynamiquement
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=val_test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),       transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"\nDataset chargé :")
print(f"  Train      : {len(train_dataset)} images")
print(f"  Validation : {len(val_dataset)} images")
print(f"  Test       : {len(test_dataset)} images")
print(f"  Classes    : {NUM_CLASSES}")

#chargement du modèle
print("\nChargement de ResNet-50 préentraîné sur ImageNet")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#gel des paramètres
for param in model.parameters():
    param.requires_grad = False

#remplacement de la couche finale
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

#entraînement de la couche finale seule
print("Phase 1 : Entraînement de la couche finale")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds    = torch.max(outputs, 1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)

    return total_loss / total, correct / total


PHASE1_EPOCHS = 5
best_val_acc  = 0.0

for epoch in range(1, PHASE1_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()
    elapsed = time.time() - t0

    print(f"  Époque {epoch}/{PHASE1_EPOCHS} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
          f"Temps: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"    → Meilleur modèle sauvegardé (val acc: {best_val_acc:.4f})")


#fine tuning pour les dernières couches

print("Phase 2 : Fine-tuning des dernières couches")

#dégel des derniers blocs et la couche finale
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FINETUNE
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS

for epoch in range(1, PHASE2_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()
    elapsed = time.time() - t0

    print(f"  Époque {epoch}/{PHASE2_EPOCHS} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
          f"Temps: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"    → Meilleur modèle sauvegardé (val acc: {best_val_acc:.4f})")

#evaluation sur le jeu de test
print("Évaluation finale : Jeu de test")

#chargement du meilleur modèle
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

class_names = train_dataset.classes
test_acc    = (all_preds == all_labels).mean()

print(f"\nAccuracy globale (test) : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"\nRapport de classification :\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

print(f"\nModèle final sauvegardé dans : {MODEL_PATH}")