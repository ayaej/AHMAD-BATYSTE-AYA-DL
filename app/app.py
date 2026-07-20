from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import json

# Configuration  page
st.set_page_config(
    page_title="Classifieur Botanique",
    page_icon="🌿",
    layout="centered",
)

# chemins et constantes
BASE_DIR = Path(__file__).resolve().parent.parent  
MODELS_DIR = BASE_DIR / "models"
MAPPING_FILE = BASE_DIR / "notebooks/flower_mapping.json"
IMAGE_SIZE = (224, 224)

MODEL_OPTIONS = {
    "Modèle 1 - model.keras": MODELS_DIR / "model.keras",
    "Modèle 2 - model_v2_augmented.keras": MODELS_DIR / "model_v2_augmented.keras",
}

# loading des ressources
@st.cache_data
def load_class_mapping():
    """Charge le dictionnaire des classes en mémoire une seule fois."""
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

CLASS_MAPPING = load_class_mapping()

@st.cache_resource
def load_model(model_path: str):
    """Charge le modèle IA en mémoire."""
    return tf.keras.models.load_model(model_path)

# fonction utils
def get_class_label(index: int) -> str:
    """Traduit l'index numérique du modèle en nom d'espèce."""
    if CLASS_MAPPING:
        return CLASS_MAPPING.get(str(index), f"Espèce inconnue (Index {index})")
    return f"Classe {index + 1}"

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prépare l'image pour le modèle MobileNetV2."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


## INTERFACE !!! 
st.title("🌿 Identificateur d'Espèces de Plantes 🌿")

st.markdown(
    """
Bienvenue dans l'application de reconnaissance botanique.

Choisissez un modèle dans la barre latérale, puis envoyez une image pour obtenir une prédiction réelle.
"""
)

# Sidebar/ Choix du modèle
st.sidebar.markdown("### Modèle IA")
selected_model_name = st.sidebar.selectbox(
    "Choisir le modèle à utiliser",
    list(MODEL_OPTIONS.keys()),
)
selected_model_path = MODEL_OPTIONS[selected_model_name]

st.sidebar.caption("Les modèles sont chargés depuis le dossier models/.")

# sécurité
if not selected_model_path.exists():
    st.error(f"Fichier introuvable : {selected_model_path.name}")
    st.stop()

st.info("Importez une image dans la case ci-dessous pour commencer l'analyse.")

# upload image
uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"],
)

# si un fichier est téléchargé, essayez de l'ouvrir comme une image
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Image sélectionnée",
            use_container_width=True,
        )

        with st.spinner(f"Analyse en cours avec {selected_model_name}..."):
            # Chargement du modèle et prédiction
            model = load_model(str(selected_model_path))
            predictions = np.asarray(model.predict(preprocess_image(image), verbose=0))
            
            if predictions.ndim > 1:
                predictions = predictions[0]

            # Top 3
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_1_index = int(top_3_indices[0])
            top_1_class = get_class_label(top_1_index)
            top_1_confidence = float(predictions[top_1_index])

        st.success(f"Modèle utilisé : {selected_model_name}")

        # Affichage du résultat principal
        st.subheader("Résultat de l'analyse")
        if top_1_confidence < 0.50:
            st.warning(
                f"**Attention :** Le modèle manque de certitude. "
                f"Il penche pour **{top_1_class}** à {top_1_confidence:.1%}."
            )
        else:
            st.success(
                f"**Prédiction principale : {top_1_class}** ({top_1_confidence:.1%} de confiance)"
            )

        # Affichage top 3
        st.markdown("### Top 3 des classes les plus probables :")
        for rank, idx in enumerate(top_3_indices, start=1):
            class_name = get_class_label(int(idx))
            confidence = float(predictions[int(idx)])

            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{rank}. {class_name}**")
            with col2:
                st.progress(confidence, text=f"{confidence:.1%}")

    except UnidentifiedImageError:
        st.error("Le fichier n'est pas une image valide.")