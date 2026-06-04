
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paramètres
SOURCE_DATASET = "../dataset/Plant_leave_diseases_dataset_without_augmentation"
DESTINATION = "../PlantVillage_split_equalemment_augmente"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
TARGET_PER_CLASS = 1000  # Nombre cible d'images par classe (modifiable)


def detect_background_mask(image):
    """
    Détection simple et précise pour fonds gris (PlantVillage).
    Un pixel de fond = faible saturation + pas une teinte de feuille.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32)

    # --- Seuil de saturation strict ---
    # Le fond gris a une saturation très basse (<40)
    # Les feuilles (vertes, jaunes, brunes) ont toujours sat > 40
    bg_mask = (s_ch < 40).astype(np.uint8) * 255

    # --- Nettoyage morphologique léger ---
    # Érosion agressive : recule le masque pour protéger les contours des feuilles
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_mask = cv2.erode(bg_mask, kernel_erode, iterations=3)

    # Fermeture légère : unifie les zones de fond fragmentées
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Garde uniquement les grandes régions connexes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask)
    min_area = image.shape[0] * image.shape[1] * 0.005  # 0.5% minimum
    clean_mask = np.zeros_like(bg_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == lbl] = 255

    return clean_mask.astype(bool)


def add_natural_background(image, mode="random", seed=None):
    """
    Remplace le fond par une texture naturelle (terre ou herbe),
    ou conserve le fond original (50% de chance).
    """
    rng = np.random.default_rng(seed)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être en BGR 3 canaux.")

    # 50% de chance de garder le fond original
    if rng.random() < 0.5:
        return image.copy()

    if mode == "random":
        mode = rng.choice(["dirt", "grass"])

    h, w, _ = image.shape

    background_mask = detect_background_mask(image)

    # Si le masque est vide ou couvre moins de 2% → on ne touche rien
    if background_mask.sum() < h * w * 0.02:
        return image.copy()

    texture = _generate_natural_texture(h, w, mode, rng)

    result = image.copy()
    result[background_mask] = texture[background_mask]
    return result


def _generate_natural_texture(h, w, mode, rng):
    """
    Génère une texture BGR imitant la terre ou l'herbe via
    plusieurs couches de bruit additionnées (Perlin-like).
    """
    # Bruit multi-échelle (simule le bruit de Perlin avec des gaussiennes)
    def multi_scale_noise(shape, scales, rng):
        noise = np.zeros(shape, dtype=np.float32)
        total_weight = 0
        for scale, weight in scales:
            # Génère un petit bruit puis l'upscale (interpolation bicubique)
            small_h = max(2, shape[0] // scale)
            small_w = max(2, shape[1] // scale)
            small   = rng.standard_normal((small_h, small_w)).astype(np.float32)
            big     = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
            noise  += big * weight
            total_weight += weight
        noise /= total_weight
        # Normalise entre 0 et 1
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise

    scales = [(2, 0.5), (4, 0.3), (8, 0.15), (16, 0.05)]
    base_noise   = multi_scale_noise((h, w), scales, rng)
    detail_noise = multi_scale_noise((h, w), [(1, 0.6), (2, 0.4)], rng)

    if mode == "dirt":
        # Terre : tons marron/ocre variés
        palettes = [
            np.array([30,  60,  90],  dtype=np.float32),   # marron foncé  (BGR)
            np.array([50,  90,  130], dtype=np.float32),   # marron moyen  (BGR)
            np.array([60,  110, 160], dtype=np.float32),   # terre ocre    (BGR)
            np.array([40,  75,  115], dtype=np.float32),   # marron rouge  (BGR)
        ]
        # Quelques taches plus claires (cailloux, argile)
        stone_mask = base_noise > 0.78
        color_idx  = (base_noise * (len(palettes) - 1)).astype(int)
        color_idx  = np.clip(color_idx, 0, len(palettes) - 1)

        texture = np.zeros((h, w, 3), dtype=np.float32)
        for idx, color in enumerate(palettes):
            mask = color_idx == idx
            texture[mask] = color

        # Variation fine avec le bruit de détail
        texture += (detail_noise[..., None] - 0.5) * 25

        # Taches claires (cailloux)
        stone_color = np.array([120, 130, 140], dtype=np.float32)
        texture[stone_mask] = stone_color + (rng.standard_normal((stone_mask.sum(), 3)) * 8).astype(np.float32)

    elif mode == "grass":
        # Herbe : tons verts variés
        palettes = [
            np.array([20,  100, 20],  dtype=np.float32),   # vert foncé   (BGR)
            np.array([30,  130, 30],  dtype=np.float32),   # vert moyen   (BGR)
            np.array([40,  160, 50],  dtype=np.float32),   # vert clair   (BGR)
            np.array([25,  80,  25],  dtype=np.float32),   # vert olive   (BGR)
        ]
        # Quelques taches sèches/jaunies
        dry_mask  = base_noise > 0.82
        color_idx = (base_noise * (len(palettes) - 1)).astype(int)
        color_idx = np.clip(color_idx, 0, len(palettes) - 1)

        texture = np.zeros((h, w, 3), dtype=np.float32)
        for idx, color in enumerate(palettes):
            mask = color_idx == idx
            texture[mask] = color

        texture += (detail_noise[..., None] - 0.5) * 20

        # Brins d'herbe sèche/jaunie
        dry_color = np.array([40, 160, 160], dtype=np.float32)
        texture[dry_mask] = dry_color + (rng.standard_normal((dry_mask.sum(), 3)) * 10).astype(np.float32)

    texture = np.clip(texture, 0, 255).astype(np.uint8)
    return texture

# Reset dossier destination
if os.path.exists(DESTINATION):
    print(f"️ Dossier existant supprimé : {DESTINATION}")
    shutil.rmtree(DESTINATION)
for subset in ["train", "validation", "test"]:
    os.makedirs(os.path.join(DESTINATION, subset), exist_ok=True)

# Récupérer toutes les classes et compter les images
class_counts = {}
class_images = {}
for class_name in os.listdir(SOURCE_DATASET):
    class_path = os.path.join(SOURCE_DATASET, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    class_counts[class_name] = len(images)
    class_images[class_name] = images

max_count = max(class_counts.values())
target_count = min(max_count, TARGET_PER_CLASS)

# Split et augmentation
for class_name, images in class_images.items():
    class_path = os.path.join(SOURCE_DATASET, class_name)
    if len(images) == 0:
        continue
    # Split
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=SEED,
        shuffle=True
    )
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
        output_class = os.path.join(DESTINATION, split_name, class_name)
        os.makedirs(output_class, exist_ok=True)
        # Copie des images originales
        for file in split_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_class, file)
            shutil.copy2(src, dst)
        # Augmentation si besoin
        n_to_add = target_count - len(split_files)
        if n_to_add > 0:
            for i in range(n_to_add):
                img_file = split_files[i % len(split_files)]
                src = os.path.join(class_path, img_file)
                img = cv2.imread(src)
                if img is None:
                    continue
                # Dans la boucle d'augmentation
                noisy_img = add_natural_background(img, mode="random", seed=i)
                aug_name  = f"aug_natural_bg_{i}_{img_file}"
                aug_path = os.path.join(output_class, aug_name)
                cv2.imwrite(aug_path, noisy_img)
    print(f"{class_name} → Train={len(train_imgs)} | Val={len(val_imgs)} | Test={len(test_imgs)} | Augmentées={max(0, target_count - len(train_imgs))}")

print("\nSéparation et augmentation terminées.")
print(f"Dataset prêt dans : {DESTINATION}")
