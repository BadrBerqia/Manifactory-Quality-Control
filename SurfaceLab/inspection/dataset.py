"""Sous-ensemble inclus de MVTec AD, catégorie cuir."""

import json

from .media import read_rgb
from .models import ROOT, fit, resize

DATA = ROOT / "data"

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


def fit_demo(backbone, catalog):
    keys = ("train", "calibration")
    sets = [[resize(read_sample(path)[0]) for path in catalog[key]] for key in keys]
    return fit(backbone, *sets)
