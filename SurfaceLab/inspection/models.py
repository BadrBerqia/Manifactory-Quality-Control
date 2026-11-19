"""Descripteurs, apprentissage sur images saines et cartes d'anomalie."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "resnet18_features.onnx"
SIZE = 256
GRID = SIZE // 8
METHODS = {
    "texture": ("Texture & couleur", "Contraste local et écarts chromatiques en Lab"),
    "padim": ("PaDiM", "Distance de Mahalanobis par position"),
    "patchcore": ("PatchCore", "Distance à la mémoire de zones saines"),
}
SCENARIOS = {
    "normal": "Image originale",
    "light": "Éclairage réduit",
    "blur": "Flou optique",
    "noise": "Bruit capteur",
}
OPENING = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# Un petit nombre de threads évite la concurrence avec le moteur ONNX.
cv.setNumThreads(2)


class Backbone:
    """ResNet-18 ImageNet tronqué après les étages 2 et 3, en inférence CPU."""

    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(WEIGHTS), options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        rng = np.random.default_rng(32)
        self.channels = [rng.choice(size, 48, replace=False) for size in (128, 256)]

    def __call__(self, rgb):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb.astype(np.float32) / 255 - mean) / std
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        levels = self.session.run(None, {self.input: tensor})
        spatial, pooled = [], []
        for level, channels in zip(levels, self.channels):
            feat = level[0, channels].transpose(1, 2, 0).copy()
            # PatchCore agrège les voisins à leur résolution d'origine.
            local = cv.blur(feat, (3, 3))
            spatial.append(cv.resize(feat[:, :, :12], (GRID, GRID)))
            pooled.append(cv.resize(local, (GRID, GRID)))
        return (
            np.concatenate(spatial, axis=-1).reshape(GRID * GRID, 24),
            np.concatenate(pooled, axis=-1).reshape(GRID * GRID, 96),
        )


def resize(rgb):
    """Normalise la grille de calcul. L'affichage conserve les proportions."""
    return cv.resize(rgb, (SIZE, SIZE), interpolation=cv.INTER_AREA)


def perturb(rgb, scenario, seed=42):
    """Les mêmes pixels perturbés sont donnés à toutes les méthodes."""
    if scenario == "light":
        return np.rint(rgb.astype(np.float32) * 0.70).astype(np.uint8)
    if scenario == "blur":
        return cv.GaussianBlur(rgb, (0, 0), 2.0)
    if scenario == "noise":
        noise = np.random.default_rng(seed).normal(0, 10, rgb.shape)
        return np.clip(np.rint(rgb + noise), 0, 255).astype(np.uint8)
    return rgb


def texture_map(rgb):
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    light = lab[:, :, 0]
    chroma = lab[:, :, 1:].copy()
    fine = np.abs(light - cv.GaussianBlur(light, (0, 0), 2.0))
    broad = np.abs(light - cv.GaussianBlur(light, (0, 0), 9.0))
    color = chroma - cv.GaussianBlur(chroma, (0, 0), 9.0)
    edge_energy = cv.GaussianBlur(fine, (0, 0), 2.0)
    texture = np.abs(edge_energy - cv.GaussianBlur(edge_energy, (0, 0), 7.0))
    return 0.55 * fine + 0.45 * broad + np.linalg.norm(color, axis=2) + 0.75 * texture


def representative_bank(features, limit=768):
    """Sous-échantillonne puis couvre les candidats par parcours du plus éloigné."""
    rng = np.random.default_rng(23)
    rows = features.reshape(-1, features.shape[-1])
    rows = rows[rng.choice(len(rows), min(5000, len(rows)), replace=False)]
    projection = (rng.standard_normal((rows.shape[1], 24)) / np.sqrt(24)).astype(np.float32)
    projected = rows @ projection
    distance = np.full(len(rows), np.inf, dtype=np.float32)
    index = 0
    representatives = []
    for _ in range(min(limit, len(rows))):
        representatives.append(index)
        delta = projected - projected[index]
        np.minimum(distance, np.einsum("ij,ij->i", delta, delta), out=distance)
        distance[index] = -1
        index = int(distance.argmax())
    return np.ascontiguousarray(rows[representatives])


@dataclass
class Profile:
    mean: np.ndarray
    precision: np.ndarray
    bank: np.ndarray
    thresholds: dict

    def learned_bytes(self, method):
        if method == "padim":
            return self.mean.nbytes + self.precision.nbytes
        return self.bank.nbytes if method == "patchcore" else 0


def fit(backbone, train, calibration):
    # Les poids CNN restent figés. Seuls les modèles de normalité sont ajustés.
    spatial, pooled = (np.stack(items) for items in zip(*(backbone(rgb) for rgb in train)))
    mean = spatial.mean(axis=0)
    centered = spatial - mean
    covariance = np.einsum("npi,npj->pij", centered, centered) / (len(train) - 1)
    # Rétraction vers la diagonale et régularisation pour les petits lots sains.
    variance = np.diagonal(covariance, axis1=1, axis2=2).copy()
    ridge = max(float(np.mean(variance)) * 0.05, 0.0001)
    covariance *= 0.55
    diagonal = np.arange(mean.shape[1])
    covariance[:, diagonal, diagonal] += 0.45 * variance + ridge
    profile = Profile(mean, np.linalg.inv(covariance), representative_bank(pooled), {})

    peaks = {key: [] for key in METHODS}
    for rgb in calibration:
        features = backbone(rgb)
        for key in METHODS:
            peaks[key].append(float(raw_map(key, rgb, profile, features).max()))
    profile.thresholds = {
        key: max(float(np.quantile(values, 0.95)) * 1.10, 0.001)
        for key, values in peaks.items()
    }
    return profile


def raw_map(method, rgb, profile, features):
    if method == "texture":
        heat = texture_map(rgb)
    else:
        spatial, pooled = features
        if method == "padim":
            delta = spatial - profile.mean
            squared = np.einsum("pi,pij,pj->p", delta, profile.precision, delta)
            distances = np.sqrt(np.maximum(0, squared))
        else:
            distances, _ = cv.batchDistance(pooled, profile.bank, cv.CV_32F, normType=cv.NORM_L2, K=1)
        heat = cv.resize(distances.reshape(GRID, GRID), (SIZE, SIZE), interpolation=cv.INTER_LINEAR)
        heat = cv.GaussianBlur(heat, (0, 0), 2.0)
    # Ouverture sur la carte continue, avant le seuil, pour écarter les pics isolés.
    return cv.morphologyEx(heat, cv.MORPH_OPEN, OPENING)


def score_all(backbone, profile, rgb):
    """Le temps autonome de chaque méthode profonde inclut l'extraction partagée."""
    start = perf_counter()
    prepared = resize(rgb)
    preparation_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    features = backbone(prepared)
    feature_ms = (perf_counter() - start) * 1000
    for method in METHODS:
        start = perf_counter()
        heat = raw_map(method, prepared, profile, features) / profile.thresholds[method]
        elapsed = preparation_ms + (perf_counter() - start) * 1000
        if method != "texture":
            elapsed += feature_ms
        yield method, heat, elapsed
