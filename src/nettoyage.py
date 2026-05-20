import os
import shutil
from sklearn.model_selection import train_test_split
#definition des paramètres
SOURCE_DATASET = "../dataset/Plant_leave_diseases_dataset_without_augmentation"
DESTINATION = "../PlantVillage_split"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

#reset au cas où
if os.path.exists(DESTINATION):
    print(f"️ Dossier existant supprimé : {DESTINATION}")
    shutil.rmtree(DESTINATION)
#creation des sous dossiers
for subset in ["train", "validation", "test"]:
    os.makedirs(os.path.join(DESTINATION, subset), exist_ok=True)
#parcours des différentes classes
for class_name in os.listdir(SOURCE_DATASET):
    class_path = os.path.join(SOURCE_DATASET, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(images) == 0:
        continue
    #split train autre
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=SEED,
        shuffle=True
    )
    #split val test
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
    #copie des fichiers
    for split_name, split_files in splits.items():
        output_class = os.path.join(DESTINATION, split_name, class_name)
        os.makedirs(output_class, exist_ok=True)
        for file in split_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_class, file)
            shutil.copy2(src, dst)
    print(
        f"{class_name} → "
        f"Train={len(train_imgs)} | "
        f"Val={len(val_imgs)} | "
        f"Test={len(test_imgs)}"
    )
print("\nSéparation terminée.")
print(f"Dataset prêt dans : {DESTINATION}")
