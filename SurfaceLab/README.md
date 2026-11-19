# SurfaceLab

SurfaceLab est une application locale en Python pour comparer trois méthodes de contrôle visuel. L’interface fonctionne dans le navigateur et les calculs s’effectuent sur le CPU. Les images importées sont traitées sur la machine.

## Lancer

Sous Windows, décompresser le ZIP, ouvrir le dossier `SurfaceLab`, puis double-cliquer sur **Demarrer.cmd**. Python 3.11 à 3.14 (64 bits) est nécessaire. La première exécution installe les dépendances dans `.venv` et nécessite Internet. Les images, les annotations et le réseau ONNX sont fournis. Les exécutions suivantes fonctionnent hors ligne.

Pour lancer manuellement depuis le dossier de l’application, avec un environnement Python dédié :

```console
python -m pip install -r requirements.txt
python run.py
```

Le navigateur s’ouvre automatiquement lorsque le serveur est prêt. L’adresse locale s’affiche dans le terminal. Lancer avec `python run.py --no-browser` pour ouvrir l’adresse manuellement. Arrêter avec **Ctrl+C** dans le terminal.

## Démonstration

1. Ouvrir l’exemple **Entaille**. Comparer la carte de chaleur des trois méthodes, puis choisir **Contours** ou **Masque**. Cliquer sur une image pour l’agrandir.
2. Activer **Afficher l’annotation** pour superposer le masque MVTec à l’image source. Observer le score, le nombre de régions, le temps CPU et l’IoU. Les masques prédits s’exportent en PNG.
3. Sélectionner **Surface saine**, puis **Éclairage réduit**. Comparer les décisions. Revenir à l’image originale et déplacer le **Seuil relatif** pour observer l’effet sur les régions.
4. Dans **Évaluation**, lancer le calcul sur 52 images et quatre conditions. Parcourir les résultats par scénario et exporter le CSV complet.
5. Pour un autre produit, créer un profil à partir de **8 à 40 photos saines et distinctes**, puis importer une photo à contrôler. Conserver le même cadrage, la même surface et un éclairage stable. Un profil personnalisé reste en mémoire pendant la session. La sélection d’un exemple MVTec réactive le profil cuir.

Formats acceptés : PNG, JPEG, WebP, BMP et TIFF. Limite par image : 16 Mo et 24 mégapixels. L’analyse utilise une grille de 256 × 256 ; l’affichage respecte les proportions. Sur les exemples annotés, les masques réels servent seulement à la comparaison visuelle et au calcul de l’IoU.

## Reproduire les résultats

```console
python -m inspection.benchmark --scenario all --threshold 1.0 --output evaluation.csv
```

Le profil cuir par défaut est calculé à partir de **209 images saines d’apprentissage et de 36 autres images saines de calibration**. Les paramètres calculés sont fournis dans `models/leather_profile.npz`. Le manifeste `data/manifest.json` identifie les références d’origine et les 52 images de test incluses. Ce test intégré utilise le même profil que l’essai à 209 références du compte rendu, sur un sous-ensemble des 124 images du rapport.

Le protocole mesure l’AUROC image, l’AUROC pixel sur 4 096 classes de scores, l’IoU moyenne des pièces défectueuses, la précision, le rappel, le taux de fausses alertes et la latence médiane et P95. La latence comprend le redimensionnement et l’inférence CPU et exclut la lecture et le rendu. La taille indiquée dans le CSV correspond aux paramètres du profil par méthode et au fichier ONNX partagé.

Chaque méthode divise sa carte d’anomalie par son seuil calibré sur les surfaces saines. Un score proche de 1,00× indique que le maximum de la carte est proche du seuil par défaut. Le score exprime un écart à la référence. Les boîtes proviennent des composantes connexes du masque.

## Organisation du code

| Élément | Rôle |
| --- | --- |
| `run.py` et `serve.py` | Démarrage local, API et gestion des profils |
| `inspection/models.py` | Texture/couleur, extracteur ResNet-18, PaDiM et PatchCore |
| `inspection/media.py` | Chargement des images, superpositions et masques |
| `inspection/dataset.py` | Lecture du lot de démonstration |
| `inspection/benchmark.py` | Mesures et évaluation reproductible en ligne de commande |
| `web/` | Interface responsive en HTML, CSS et JavaScript |
| `data/` et `models/` | Images, masques et poids ONNX utilisés par l’application |

Le compte rendu explique les variantes des méthodes, les résultats et les limites du prototype. Les droits des [données](data/README.md) et des [poids préentraînés](models/NOTICE.md) sont documentés avec les ressources.
