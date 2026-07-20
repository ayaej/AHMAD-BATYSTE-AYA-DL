import streamlit as st
from PIL import Image, UnidentifiedImageError

# Configuration
st.set_page_config(
    page_title="Classifieur Botanique",
    page_icon="🌿",
    layout="centered"
)

# Accueil
st.title("🌿 Identificateur d'Espèces de Plantes 🌿")

st.markdown("""
Bienvenue dans notre application de reconnaissance botanique !

Cette application permettra d'identifier une espèce de fleur
à partir d'une image.
""")

st.info("Importez une image dans la case ci-dessous pour commencer l'analyse.")

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)

# si un fichier est téléchargé, essayez de l'ouvrir comme une image
if uploaded_file:

    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Image sélectionnée",
            use_container_width=True
        )

        st.success("Image chargée avec succès !")

    except UnidentifiedImageError:
        st.error("Le fichier n'est pas une image valide.")