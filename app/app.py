import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np

# début de la partie mock-up
# Simulation statique 
CLASS_NAMES = {
    "0": "Rose rouge",
    "1": "Tulipe jaune",
    "2": "Tournesol",
    "3": "Orchidée",
    "4": "Marguerite",
    "5": "Lys blanc"
}

def mock_predict(mode):
    """Génère de fausses probabilités pour simuler le comportement du modèle"""
    if mode == "haute_confiance":
        # Le modèle est sûr de lui (85% sur Tournesol, 10% Rose, 5% Marguerite)
        return np.array([0.10, 0.00, 0.85, 0.00, 0.05, 0.00])
    else:
        # Le modèle hésite (Softmax trap : 40% Tulipe, 35% Orchidée, 25% Lys)
        return np.array([0.00, 0.40, 0.00, 0.35, 0.00, 0.25])

# Side bar pour simuler le mode de test
st.sidebar.markdown("### Mode Test (Mock-up)")
st.sidebar.write("Simulez le comportement du modèle pour tester l'interface :")
test_mode = st.sidebar.radio(
    "Niveau de confiance :", 
    ["Haute confiance (>50%)", "Basse confiance (<50%)"]
)
mode_arg = "haute_confiance" if test_mode == "Haute confiance (>50%)" else "basse_confiance"
# ---- fin partie mock-up ----

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

        with st.spinner("Analyse en cours... (Simulation)"):
            # mocku^p
            predictions = mock_predict(mode=mode_arg)
            
            # Récupération du Top-3
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_1_index = top_3_indices[0]
            top_1_class = CLASS_NAMES[str(top_1_index)]
            top_1_confidence = predictions[top_1_index]

        # Affichage des résultats
        st.subheader("Résultat de l'analyse")
        if top_1_confidence < 0.50:
            st.warning(
                f"**Attention :** Le modèle manque de certitude. "
                f"Il penche pour **{top_1_class}** à {top_1_confidence:.1%}, "
                f"mais l'image est peut-être atypique, floue, ou l'espèce est hors dataset."
            )
        else:
            st.success(f"**Prédiction principale : {top_1_class}** ({top_1_confidence:.1%} de confiance)")

        # Affichage du Top-3
        st.markdown("### Top 3 des espèces les plus probables :")
        for i, idx in enumerate(top_3_indices):
            class_name = CLASS_NAMES[str(idx)]
            confidence = float(predictions[idx])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{i+1}. {class_name}**")
            with col2:
                st.progress(confidence, text=f"{confidence:.1%}")

    except UnidentifiedImageError:
        st.error("Le fichier n'est pas une image valide.")

