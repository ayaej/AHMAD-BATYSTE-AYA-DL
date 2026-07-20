# AHMAD-BATYSTE-AYA-DL

Classifieur d'espèces de plantes — projet de deep learning en groupe (Ahmad, Batyste, Aya).

## 1. Problème

À partir d'une photo, identifier l'espèce de plante parmi 100+ classes. Cas d'usage : application mobile
pour jardiniers ou botanistes amateurs qui veulent identifier une plante en un clic.

## 2. Dataset

- **Choix retenu** : [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) — 102 classes
  de fleurs, ~8 000 images, disponible directement via `tensorflow_datasets` (`oxford_flowers102`), sans
  dépendance à l'API Kaggle. Choisi pour sa fiabilité de chargement en début de journée.
- **Alternative envisagée** : PlantCLEF 2022 (Kaggle) — beaucoup plus de classes et d'images, mais dataset
  plus lourd et plus long à charger/préparer. Option de repli si le groupe veut pousser plus loin après
  convergence sur Oxford 102 Flowers.
- **Répartition train/val/test** : à confirmer après inspection de la distribution des classes (le split
  officiel du dataset sera utilisé comme point de départ).
- **Format** : images RGB, tailles variables, redimensionnées à 224x224 pour correspondre à l'entrée de
  MobileNetV2.

## 3. Architecture

- **Approche** : transfer learning avec **MobileNetV2** pré-entraîné sur ImageNet, couches de base gelées,
  fine-tuning de la tête de classification.
- **Justification** : le dataset (quelques milliers d'images pour 102 classes) est trop petit pour entraîner
  un CNN from scratch sur des images 224x224. MobileNetV2 a déjà appris des features génériques (bords,
  textures, formes) sur 1.2M d'images ImageNet ; seule la tête finale doit être spécialisée sur les espèces
  de plantes.
- **Schéma des couches** : à ajouter (Excalidraw) une fois la tête de classification finalisée.

## 4. Plan d'entraînement

- **Plateforme** : Google Colab (GPU T4) ou Kaggle Notebooks.
- **Optimizer** : Adam, learning rate initial 0.001.
- **Loss** : categorical crossentropy.
- **Métrique cible** : accuracy (classification multi-classes).
- **Baseline aléatoire** : 1/102 ≈ 0.98% — seuil minimum à dépasser pour prouver que le modèle apprend.
  Objectif visé : ≥ 70% avant de déclarer victoire.
- **Epochs** : à démarrer bas (5-10), augmenter si la loss de validation continue de descendre.

## 5. Répartition des rôles

| Rôle | Responsable |
|---|---|
| Chargement et inspection des données | Ahmad |
| Modèle (transfer learning, entraînement) | Ahmad |
| WebApp (Streamlit) | à définir |
| README / préparation soutenance | à définir |

Le lead du groupe (point de contact, responsable du repo) : à désigner.

Les rôles peuvent se croiser en cours de journée — cette table sera mise à jour au fil de l'avancement.

## 6. Questions ouvertes

- Garder Oxford 102 Flowers ou basculer sur PlantCLEF 2022 si le temps le permet ?
- Nombre d'epochs final : décidé après observation des premières courbes loss train/val.
- Faut-il de la data augmentation dès le départ, ou seulement si overfitting observé ?

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Entraînement : voir `notebooks/` (à venir).

Lancer la WebApp une fois le modèle entraîné et sauvegardé dans `models/model.keras` :

```bash
streamlit run app.py
```

## Résultats

_À compléter après entraînement (accuracy, loss, comparaison au baseline aléatoire)._

## Limites

_À compléter après les tests adversariaux (checkpoint 9)._

## Tests adversariaux

_À compléter au checkpoint 9 — tableau input / comportement observé / explication._
