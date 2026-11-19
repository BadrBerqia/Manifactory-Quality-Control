"""Sous-ensemble inclus de MVTec AD, catégorie cuir."""

import json

import numpy as np

from .media import read_rgb
from .models import METHODS, ROOT, Profile

DATA = ROOT / "data"
DEMO_PROFILE = ROOT / "models" / "leather_profile.npz"

LABELS = {
    "good": "Surface saine",
    "cut": "Entaille",
    "color": "Tache de couleur",
    "fold": "Pli",
    "glue": "Trace de colle",
    "poke": "Perforation",
}


def read_catalog():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def read_sample(relative_path):
    return read_rgb((DATA / relative_path).read_bytes())


def load_demo_profile():
    # Profil du rapport, pré-calculé pour démarrer sans les 245 photos saines.
    with np.load(DEMO_PROFILE, allow_pickle=False) as weights:
        thresholds = {key: float(weights[f"{key}_threshold"]) for key in METHODS}
        return Profile(weights["mean"], weights["precision"], weights["bank"], thresholds)
