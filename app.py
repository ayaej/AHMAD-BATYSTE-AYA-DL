"""WebApp Streamlit - Classifieur d'espèces de plantes.

Squelette à compléter au checkpoint 8 (déploiement), une fois le modèle entraîné et sauvegardé.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/model.keras"
CLASS_NAMES = []  # TODO: remplir avec les noms des classes du dataset choisi
IMAGE_SIZE = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("Classifieur d'espèces de plantes")
st.write("Uploadez une photo de plante pour obtenir une prédiction d'espèce.")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model()
    image = Image.open(uploaded_file).convert("RGB").resize(IMAGE_SIZE)
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    st.image(image, caption="Image uploadée", use_column_width=True)
    st.success(f"Prédiction : {predicted_class} ({confidence:.1%} de confiance)")
else:
    st.info("Aucune image sélectionnée.")
