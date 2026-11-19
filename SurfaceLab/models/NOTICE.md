# Extracteur ResNet-18

`resnet18_features.onnx` est un export ONNX (opset 17) de [ResNet-18 TorchVision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) avec les poids [ImageNet-1K V1](https://download.pytorch.org/models/resnet18-f37072fd.pth). Le réseau a été tronqué après `layer3` et fournit les activations de `layer2` et `layer3` pour une entrée RGB normalisée au format NCHW et de taille 1 × 3 × 256 × 256. Les poids d’origine sont figés. Le fichier est livré avec l’[avis de licence TorchVision](LICENSE-TorchVision.txt).
