import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score, accuracy_score
import time

#definition des paramètres
DATA_DIR        = "PlantVillage_split"
RESNET_PATH     = "models/resnet50_plantvillage.pth"
EFFICIENT_PATH  = "models/efficientnet_b0_plantvillage.pth"
BATCH_SIZE      = 32

#choix du device mps ou cuda ou cpu
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print(f"Appareil utilisé : {DEVICE}")

#transformation val/test uniquement
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#chargement du dataset de test
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

NUM_CLASSES  = len(test_dataset.classes)
class_names  = test_dataset.classes

print(f"\nJeu de test : {len(test_dataset)} images | {NUM_CLASSES} classes")


def evaluate(model):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    elapsed = time.time() - t0
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc     = accuracy_score(all_labels, all_preds)
    f1_mac  = f1_score(all_labels, all_preds, average="macro")
    f1_wei  = f1_score(all_labels, all_preds, average="weighted")

    return all_preds, all_labels, acc, f1_mac, f1_wei, elapsed


#chargement et évaluation de ResNet-50
print("\nChargement de ResNet-50")
resnet = models.resnet50(weights=None)
resnet.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(resnet.fc.in_features, NUM_CLASSES))
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
resnet = resnet.to(DEVICE)

resnet_preds, resnet_labels, resnet_acc, resnet_f1_mac, resnet_f1_wei, resnet_time = evaluate(resnet)

print(f"  Accuracy          : {resnet_acc:.4f} ({resnet_acc*100:.2f}%)")
print(f"  F1-score macro    : {resnet_f1_mac:.4f}")
print(f"  F1-score weighted : {resnet_f1_wei:.4f}")
print(f"  Temps d'inférence : {resnet_time:.1f}s ({resnet_time/len(test_dataset)*1000:.1f}ms/image)")

#chargement et évaluation de EfficientNet-B0
print("\nChargement de EfficientNet-B0")
efficient = models.efficientnet_b0(weights=None)
efficient.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(efficient.classifier[1].in_features, NUM_CLASSES))
efficient.load_state_dict(torch.load(EFFICIENT_PATH, map_location=DEVICE))
efficient = efficient.to(DEVICE)

eff_preds, eff_labels, eff_acc, eff_f1_mac, eff_f1_wei, eff_time = evaluate(efficient)

print(f"  Accuracy          : {eff_acc:.4f} ({eff_acc*100:.2f}%)")
print(f"  F1-score macro    : {eff_f1_mac:.4f}")
print(f"  F1-score weighted : {eff_f1_wei:.4f}")
print(f"  Temps d'inférence : {eff_time:.1f}s ({eff_time/len(test_dataset)*1000:.1f}ms/image)")

#comparaison directe
print("\nComparaison")
print(f"  {'Métrique':<25} {'ResNet-50':>12} {'EfficientNet-B0':>16}")
print(f"  {'-'*55}")
print(f"  {'Accuracy':<25} {resnet_acc*100:>11.2f}% {eff_acc*100:>15.2f}%")
print(f"  {'F1-score macro':<25} {resnet_f1_mac:>12.4f} {eff_f1_mac:>16.4f}")
print(f"  {'F1-score weighted':<25} {resnet_f1_wei:>12.4f} {eff_f1_wei:>16.4f}")
print(f"  {'Temps inférence (s)':<25} {resnet_time:>12.1f} {eff_time:>16.1f}")
print(f"  {'Nb paramètres':<25} {'~25.6M':>12} {'~5.3M':>16}")

#rapport de classification par classe pour chaque modèle
print("\nRapport par classe — ResNet-50")
print(classification_report(resnet_labels, resnet_preds, target_names=class_names, digits=4))

print("\nRapport par classe — EfficientNet-B0")
print(classification_report(eff_labels, eff_preds, target_names=class_names, digits=4))
