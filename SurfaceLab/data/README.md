# Données de démonstration

Images et annotations de la catégorie **leather** de **MVTec AD**, © MVTec Software GmbH, sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Le [texte de licence local](LICENSE-MVTec.txt) est inclus. Source officielle et conditions d’utilisation sur la [page du jeu de données](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Les fichiers d’origine ont été récupérés depuis le [miroir foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad).

Le sous-ensemble retient 36 exemples sains de `train/good`, 12 exemples de `test/good` et 8 images de chacune des 5 familles de défauts avec leur masque. Dans chaque dossier, les noms triés sont échantillonnés à intervalles réguliers en incluant les deux extrémités. Une image saine d’entraînement sur trois est réservée à la calibration. Les chemins exacts figurent dans `manifest.json`.

Les images ont été réduites à 640 px au plus par dimension et converties en JPEG de qualité 95. Les masques binaires ont été redimensionnés au plus proche voisin et restent en PNG. Ces adaptations sont redistribuées sous la licence du jeu d’origine pour ce projet pédagogique.

> P. Bergmann, M. Fauser, D. Sattlegger et C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR, 2019. [Article](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD--A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html).
