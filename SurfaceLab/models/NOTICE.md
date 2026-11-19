# Extracteur ResNet-18

`resnet18_features.onnx` est un export ONNX (opset 17) de [ResNet-18 TorchVision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) avec les poids [ImageNet-1K V1](https://download.pytorch.org/models/resnet18-f37072fd.pth). Le réseau a été tronqué après `layer3` et fournit les activations de `layer2` et `layer3` pour une entrée RGB normalisée au format NCHW et de taille 1 × 3 × 256 × 256. Les poids d’origine sont figés. Le fichier est livré avec l’[avis de licence TorchVision](LICENSE-TorchVision.txt).

## Profil cuir

`leather_profile.npz` contient la moyenne et la précision par position de PaDiM, la mémoire de 768 zones de PatchCore et les trois seuils de décision. C’est le profil à 209 références d’apprentissage et 36 références de calibration mesuré dans le compte rendu. Les images proviennent de MVTec AD, catégorie `leather`. Le [manifeste](../data/manifest.json) en identifie la partition exacte et la révision du miroir. Les paramètres dérivés des données sont fournis pour un usage pédagogique sous [CC BY-NC-SA 4.0](../data/LICENSE-MVTec.txt).
