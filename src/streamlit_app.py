import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

IMG_SIZE = 224
SEUIL_CONFIANCE = 0.60


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "PlantVillage_split"
    model_file = base_dir / model_path

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder(str(data_dir / "train"))
    class_names = dataset.classes
    num_classes = len(class_names)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes))
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, class_names, device


def compute_gradcam(model, tensor, target_class, device):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    try:
        output = model(tensor)
        model.zero_grad()
        output[0, target_class].backward()
    finally:
        handle_f.remove()
        handle_b.remove()

    act = activations[0].squeeze(0).detach().cpu().numpy()
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
    cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_np = np.array(original_img.convert("RGB"))
    blended = cv2.addWeighted(original_np, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(blended)


def predict_image(model, transform, class_names, image_path, device):
    original = Image.open(image_path).convert("RGB")
    tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    confidence = float(conf.item())
    class_name = class_names[pred_idx.item()].replace("___", " — ").replace("_", " ")

    tensor_grad = transform(original).unsqueeze(0).to(device)
    tensor_grad.requires_grad_(True)
    cam = compute_gradcam(model, tensor_grad, pred_idx.item(), device)
    heatmap_img = make_heatmap(original, cam)

    return original, heatmap_img, class_name, confidence


def render_app(model_path: str, title: str, accent: str):
    st.set_page_config(page_title=title, page_icon="🌿", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(135deg, #0f172a 0%, #14532d 45%, #1f2937 100%); color: #eff6ff; }}
        div[data-testid="stSidebar"] {{ background-color: #111827; }}
        .stButton > button {{ background: linear-gradient(135deg, {accent} 0%, #22c55e 100%); color: white; border: none; border-radius: 0.75rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌱 AgroScan AI")
    st.caption("Diagnostic visuel des feuilles avec analyse Grad-CAM et score de confiance")

    with st.sidebar:
        st.header("Options")
        st.info("Choisissez un modèle et une image pour lancer l’analyse.")
        st.metric("Seuil de confiance", f"{SEUIL_CONFIANCE:.0%}")
        st.caption("Si la confiance est inférieure au seuil, un message d’alerte s’affiche pour guider la décision.")

    model, transform, class_names, device = load_model(model_path)

    uploaded_file = st.file_uploader("Importer une photo de feuille", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = Path(tmp_file.name)

        try:
            with st.spinner("Analyse en cours…"):
                original, heatmap_img, class_name, confidence = predict_image(model, transform, class_names, temp_path, device)
        finally:
            temp_path.unlink(missing_ok=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Image originale")
            st.image(original, use_column_width=True)

        with col_right:
            st.subheader("Zone suspecte (Grad-CAM)")
            st.image(heatmap_img, use_column_width=True)

        st.divider()
        st.subheader("Résumé du diagnostic")

        result_col, confidence_col = st.columns([2, 1])
        with result_col:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 18px; padding: 16px 18px; margin-bottom: 12px;">
                    <div style="font-size: 0.92rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.16em;">Diagnostic</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #ecfccb; line-height: 1.35; white-space: normal; overflow-wrap: anywhere;">{class_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with confidence_col:
            st.markdown(
                f"""
                <div style="background: rgba(20, 83, 45, 0.92); border: 1px solid rgba(134, 239, 172, 0.35); border-radius: 18px; padding: 16px 18px; margin-bottom: 12px;">
                    <div style="font-size: 0.92rem; color: #dcfce7; text-transform: uppercase; letter-spacing: 0.16em;">Confiance</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #f0fdf4;">{confidence * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        status_col = st.container()
        with status_col:
            if confidence < SEUIL_CONFIANCE:
                st.warning("Score faible — une vérification humaine est recommandée pour confirmer ce diagnostic.")
            else:
                st.success("Analyse terminée — la prédiction semble cohérente avec le niveau de confiance observé.")

        with st.expander("Pourquoi cette zone est mise en évidence ?"):
            st.write("La carte Grad-CAM met en surbrillance les régions de l’image qui ont le plus influencé la décision du modèle.")
    else:
        st.info("Ajoutez une image pour démarrer le diagnostic.")
        placeholder = np.zeros((320, 480, 3), dtype=np.uint8)
        placeholder[:, :, 0] = 34
        placeholder[:, :, 1] = 197
        placeholder[:, :, 2] = 94
        st.image(placeholder, caption="Interface prête à analyser une feuille", use_column_width=True)
