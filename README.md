# AgroScan AI 

Projet réalisé dans le cadre du cours **8INF934 : Atelier pratique en intelligence artificielle I**  
Université du Québec à Chicoutimi (UQAC)

**Étudiantes :** Charlotte Chanudet & Mahaut Galice  
**Encadrant :** Julien Maitre

---

## Description

AgroScan AI est un assistant de diagnostic visuel.
À partir d'une photo de feuille, l'application identifie si la plante est saine ou malade,
retourne un score de confiance et met en évidence la zone suspecte via une carte Grad-CAM.

---

## Structure du projet

```
AgroScan/
│
├── dataset/                        # Dataset original PlantVillage (non versionné car trop lourd)
├── PlantVillage_split/             # Dataset splitté 70/15/15 (non versionné car trop lourd)
│
├── models/                         # Modèles entraînés sauvegardés
│   ├── resnet50_plantvillage.pth
│   └── efficientnet_b0_plantvillage.pth
│
├── Rapport et présentations/       # Documents LaTeX, rapports, PDF, diaporamas
│
├── src/
│   ├── nettoyage.py
│   ├── train_resnet.py
│   ├── train_efficientNet.py
│   └── visudata.ipynb
│
└── README.md

```

> **Note :** les dossiers `dataset/` et `PlantVillage_split/` ne sont pas versionnés sur Git
> car ils sont trop volumineux. Voir la section **Installation** pour les recréer.

---

## Installation

### Dépendances

```bash
pip install torch torchvision scikit-learn numpy
```

### Préparer le dataset

1. Télécharger PlantVillage via TensorFlow Datasets :
   https://www.tensorflow.org/datasets/catalog/plant_village

2. Placer le dossier téléchargé dans `dataset/`

3. Lancer le script de split :

```bash
python nettoyage.py
```

Cela créera le dossier `PlantVillage_split/` avec la structure `train/`, `validation/`, `test/`.

---

## Entraînement des modèles

Les deux scripts suivent la même structure en deux phases :

- **Phase 1** — seule la couche de classification finale est entraînée (poids du backbone gelés)
- **Phase 2** — fine-tuning des derniers blocs convolutionnels avec un taux d'apprentissage réduit

À chaque époque, la loss et l'accuracy sont affichées pour le train et la validation.
Le meilleur modèle (meilleure val accuracy) est sauvegardé automatiquement dans `models/`.

### ResNet-50

```bash
python train_resnet.py
```

Modèle sauvegardé : `models/resnet50_plantvillage.pth`

### EfficientNet-B0

```bash
python train_efficientNet.py
```

Modèle sauvegardé : `models/efficientnet_b0_plantvillage.pth`

---
