# Données de démonstration

Images et annotations de la catégorie **leather** de **MVTec AD**, © MVTec Software GmbH, sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Le [texte de licence local](LICENSE-MVTec.txt) est inclus. Source officielle et conditions d’utilisation sur la [page du jeu de données](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Les fichiers d’origine ont été récupérés depuis le [miroir foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad).

Le lot de démonstration inclut 12 exemples de `test/good` et 8 images de chacune des 5 familles de défauts avec leur masque. Dans chaque dossier de test, les noms triés sont échantillonnés à intervalles réguliers en incluant les deux extrémités.

Le manifeste indique aussi les 209 images d’apprentissage et les 36 images de calibration du profil par défaut. Ces chemins désignent les PNG du jeu d’origine. Les 24 références du profil de comparaison du rapport figurent dans `study_baseline_train`. Le profil par défaut est distribué sous forme de paramètres pré-calculés dans le dossier `models/` pour limiter la taille de l’application.

Les images ont été réduites à 640 px au plus par dimension et converties en JPEG de qualité 95. Les masques binaires ont été redimensionnés au plus proche voisin et restent en PNG. Ces adaptations sont redistribuées sous la licence du jeu d’origine pour ce projet pédagogique.

> P. Bergmann, M. Fauser, D. Sattlegger et C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR, 2019. [Article](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD--A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html).
