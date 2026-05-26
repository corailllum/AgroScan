import os
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models, datasets

#paramètres
MODEL_PATH  = "models/resnet50_plantvillage.pth"
DATA_DIR    = "PlantVillage_split"
IMG_SIZE    = 224
SEUIL_CONFIANCE = 0.60

#choix du device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

#récupération des noms de classes depuis le dataset
dataset     = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
class_names = dataset.classes
NUM_CLASSES = len(class_names)

#chargement du modèle
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#transformation de l'image
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def compute_gradcam(tensor, target_class):
    #calcul de la carte GradCAM sur la dernière couche convolutionnelle
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    output = model(tensor)
    model.zero_grad()
    output[0, target_class].backward()

    handle_f.remove()
    handle_b.remove()

    act  = activations[0].squeeze(0).detach().cpu().numpy()
    grad = gradients[0].squeeze(0).detach().cpu().numpy()
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def make_heatmap(original_img, cam):
    #superposition de la carte GradCAM sur l'image originale
    import cv2
    cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_np = np.array(original_img.convert("RGB"))
    blended = cv2.addWeighted(original_np, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(blended)


def predict(image_path):
    #inférence et génération de la GradCAM
    original = Image.open(image_path).convert("RGB")
    tensor   = transform(original).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    confidence = conf.item()
    class_name = class_names[pred_idx.item()].replace("___", " — ").replace("_", " ")

    #gradcam nécessite les gradients
    tensor_grad = transform(original).unsqueeze(0).to(DEVICE)
    tensor_grad.requires_grad_(True)
    cam = compute_gradcam(tensor_grad, pred_idx.item())
    heatmap_img = make_heatmap(original, cam)

    return original, heatmap_img, class_name, confidence


def choisir_image():
    #ouverture du navigateur de fichiers
    path = filedialog.askopenfilename(
        title="Choisir une image de feuille",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    original, heatmap_img, class_name, confidence = predict(path)

    #taille dynamique des images selon la fenêtre
    img_w = max(350, (root.winfo_width() - 320) // 2 - 30)
    img_h = max(350, root.winfo_height() - 200)

    #affichage de l'image originale
    img_orig = original.resize((img_w, img_h))
    photo_orig = ImageTk.PhotoImage(img_orig)
    label_img_orig.configure(image=photo_orig, width=img_w, height=img_h)
    label_img_orig.image = photo_orig

    #affichage de la GradCAM
    img_heat = heatmap_img.resize((img_w, img_h))
    photo_heat = ImageTk.PhotoImage(img_heat)
    label_img_heat.configure(image=photo_heat, width=img_w, height=img_h)
    label_img_heat.image = photo_heat

    #affichage du résultat
    label_classe.configure(text=class_name)
    label_confiance.configure(text=f"{confidence*100:.1f}%")

    if confidence < SEUIL_CONFIANCE:
        frame_resultat.configure(bg="#f5c518")
        label_avertissement.configure(
            text="⚠ Score faible — consultez un spécialiste",
            bg="#f5c518", fg="#333333"
        )
    else:
        frame_resultat.configure(bg="#e8f5e9")
        label_avertissement.configure(text="", bg="#e8f5e9")


#interface principale
root = tk.Tk()
root.title("AgroScan AI — Diagnostic visuel des plantes")
root.configure(bg="#f5f5f5")
root.resizable(True, True)

#taille initiale proche du plein écran
largeur  = root.winfo_screenwidth()
hauteur  = root.winfo_screenheight()
root.geometry(f"{int(largeur * 0.9)}x{int(hauteur * 0.9)}+{int(largeur * 0.05)}+{int(hauteur * 0.05)}")

#configuration des colonnes et lignes pour que tout s'étire
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=0)
root.rowconfigure(3, weight=1)

font_titre  = font.Font(family="Helvetica", size=18, weight="bold")
font_label  = font.Font(family="Helvetica", size=12)
font_result = font.Font(family="Helvetica", size=14, weight="bold")

#titre
tk.Label(root, text="AgroScan AI", font=font_titre, bg="#2e7d32", fg="white",
         padx=20, pady=14).grid(row=0, column=0, columnspan=3, sticky="ew")

#bouton import
tk.Button(root, text="Choisir une image", command=choisir_image,
          font=font_label, bg="#2e7d32", fg="#2e7d32",
          padx=20, pady=10, relief="flat", cursor="hand2"
          ).grid(row=1, column=0, columnspan=3, pady=20)

#labels images
tk.Label(root, text="Image originale", font=font_label, bg="#f5f5f5", fg="#2e7d32"
         ).grid(row=2, column=0, padx=20)
tk.Label(root, text="Carte GradCAM", font=font_label, bg="#f5f5f5", fg="#2e7d32"
         ).grid(row=2, column=1, padx=20)

#placeholders images : s'étirent avec la fenêtre
label_img_orig = tk.Label(root, bg="#cccccc")
label_img_orig.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

label_img_heat = tk.Label(root, bg="#cccccc")
label_img_heat.grid(row=3, column=1, padx=20, pady=10, sticky="nsew")

#encadré résultat fixe à droite
frame_resultat = tk.Frame(root, bg="#e8f5e9", padx=24, pady=24,
                           relief="solid", bd=1, width=260)
frame_resultat.grid(row=3, column=2, padx=20, pady=10, sticky="ns")
frame_resultat.grid_propagate(False)

tk.Label(frame_resultat, text="Diagnostic", font=font_label,
         bg="#e8f5e9", fg="#555555").pack()
tk.Label(frame_resultat, text="─" * 20, bg="#e8f5e9", fg="#aaaaaa").pack()

label_classe = tk.Label(frame_resultat, text="—", font=font_result,
                         bg="#e8f5e9", fg="#1b5e20", wraplength=220, justify="center")
label_classe.pack(pady=14)

tk.Label(frame_resultat, text="Confiance", font=font_label,
         bg="#e8f5e9", fg="#555555").pack()

label_confiance = tk.Label(frame_resultat, text="—", font=font_result,
                             bg="#e8f5e9", fg="#1b5e20")
label_confiance.pack(pady=8)

label_avertissement = tk.Label(frame_resultat, text="", font=font_label,
                                 bg="#e8f5e9", fg="#333333", wraplength=220, justify="center")
label_avertissement.pack(pady=10)

root.mainloop()
