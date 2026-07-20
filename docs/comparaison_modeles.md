## Comparaison des modèles

| Config | Accuracy test | Baseline aléatoire | Approche |
|---|---|---|---|
| v1 (Ahmad) | 76.68% | 0.98% | MobileNetV2, couches gelées, pas d'augmentation |
| v2 (Aya) | 54.53% | 0.98% | MobileNetV2, fine-tuning 30 couches dégelées + augmentation, dès le départ |

**Modèle retenu pour la production : v1**

**Analyse** : la v2 reste en dessous de la v1 malgré le fine-tuning et l'augmentation. 
Cause probable : dégeler les couches pré-entraînées en même temps qu'on entraîne une 
tête de classification encore aléatoire génère des gradients instables qui dégradent 
les features ImageNet déjà apprises. Une approche en deux phases (entraîner la tête 
seule d'abord, puis fine-tuner avec un learning rate très bas) aurait probablement 
donné un meilleur résultat, mais demande plus de temps d'entraînement.