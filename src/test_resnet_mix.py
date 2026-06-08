import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_mix_plantdoc.pth"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "PlantVillage_split_equalemment_augmente"
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device :", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

NUM_CLASSES = len(test_dataset.classes)

print("Classes :", NUM_CLASSES)
print("Images test :", len(test_dataset))

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

print("Modèle chargé.")

# ======================
# INFERENCE
# ======================

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)



acc = accuracy_score(all_labels, all_preds)

print("ACCURACY")
print(f"{acc*100:.2f}%")

print("CLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes,
        digits=4
    )
)


cm = confusion_matrix(
    all_labels,
    all_preds
)

fig, ax = plt.subplots(
    figsize=(18, 18)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "confusion_matrix_mix.png"
    ),
    dpi=300
)

plt.show()

print("\nMatrice sauvegardée : confusion_matrix_mix.png")
