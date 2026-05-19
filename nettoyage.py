
import os
import shutil
from sklearn.model_selection import train_test_split

# ======================
# PARAMÈTRES
# ======================

SOURCE_DATASET = "dataset\Plant_leave_diseases_dataset_without_augmentation"   # dossier original
DESTINATION = "PlantVillage_split" # dossier de sortie

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# ======================
# CRÉATION DOSSIERS
# ======================

for subset in ["train", "validation", "test"]:
    os.makedirs(os.path.join(DESTINATION, subset), exist_ok=True)

# ======================
# PARCOURS DES CLASSES
# ======================

for class_name in os.listdir(SOURCE_DATASET):

    class_path = os.path.join(SOURCE_DATASET, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ]

    if len(images) == 0:
        continue

    # Train = 70%
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1-TRAIN_RATIO),
        random_state=SEED,
        shuffle=True
    )

    # Validation/Test = 15/15
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=SEED,
        shuffle=True
    )

    splits = {
        "train": train_imgs,
        "validation": val_imgs,
        "test": test_imgs
    }

    for split_name, split_files in splits.items():

        output_class = os.path.join(
            DESTINATION,
            split_name,
            class_name
        )

        os.makedirs(output_class, exist_ok=True)

        for file in split_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_class, file)

            shutil.copy2(src, dst)

    print(
        f"{class_name} : "
        f"Train={len(train_imgs)} | "
        f"Validation={len(val_imgs)} | "
        f"Test={len(test_imgs)}"
    )

print("\nSéparation terminée.")
print(f"Données enregistrées dans : {DESTINATION}")
