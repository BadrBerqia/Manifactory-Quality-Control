"""Évaluation reproductible sur le lot de test local et ses perturbations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import DATA, fit_demo, read_catalog, read_sample
from .media import truth_mask
from .models import METHODS, SCENARIOS, WEIGHTS, Backbone, perturb, score_all

PIXEL_BINS = 4096


def auc_counts(positive, negative):
    """Probabilité qu'un score positif dépasse un négatif, demi-crédit aux ex æquo."""
    total = float(positive.sum()) * float(negative.sum())
    if total == 0:
        return None
    return float(positive @ (negative.cumsum() - 0.5 * negative)) / total


def image_auc(scores, labels):
    _, bins = np.unique(scores, return_inverse=True)
    return auc_counts(
        np.bincount(bins[labels], minlength=len(scores)),
        np.bincount(bins[~labels], minlength=len(scores)),
    )


class Statistics:
    def __init__(self):
        self.scores = []
        self.times = []
        self.labels = []
        self.iou = []
        self.pixel_pos = np.zeros(PIXEL_BINS, dtype=np.int64)
        self.pixel_neg = np.zeros(PIXEL_BINS, dtype=np.int64)

    def add(self, heat, elapsed, defective, truth, threshold):
        self.scores.append(float(heat.max()))
        self.times.append(elapsed)
        self.labels.append(defective)
        mask = heat >= threshold
        if defective:
            self.iou.append(np.count_nonzero(mask & truth) / np.count_nonzero(mask | truth))
        # Projection monotone des scores sur [0,1[ pour une AUROC pixel en mémoire bornée.
        bins = (heat / (1 + heat) * (PIXEL_BINS - 1)).astype(np.int32)
        self.pixel_pos += np.bincount(bins[truth], minlength=PIXEL_BINS)
        self.pixel_neg += np.bincount(bins[~truth], minlength=PIXEL_BINS)

    def summarize(self, threshold):
        labels = np.asarray(self.labels)
        scores = np.asarray(self.scores)
        flag = scores >= threshold
        tp = int((flag & labels).sum())
        fp = int((flag & ~labels).sum())
        fn = int((~flag & labels).sum())
        tn = int((~flag & ~labels).sum())
        return {
            "image_auc": image_auc(scores, labels),
            "pixel_auc": auc_counts(self.pixel_pos, self.pixel_neg),
            "iou": float(np.mean(self.iou)),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn),
            "fpr": fp / (fp + tn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "median_ms": float(np.median(self.times)),
            "p95_ms": float(np.quantile(self.times, 0.95)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }


def evaluate(backbone, profile, catalog, scenarios, threshold):
    examples = catalog["test"]
    prepared = [
        (
            read_sample(item["image"])[0],
            item["kind"] != "good",
            truth_mask(DATA / item["mask"] if "mask" in item else None),
        )
        for item in examples
    ]
    rows = []
    for scenario in scenarios:
        stats = {method: Statistics() for method in METHODS}
        for index, (rgb, defective, truth) in enumerate(prepared):
            variant = perturb(rgb, scenario, seed=42 + index)
            for method, heat, elapsed in score_all(backbone, profile, variant):
                stats[method].add(heat, elapsed, defective, truth, threshold)
        for method, result in stats.items():
            rows.append({
                "scenario": scenario,
                "method": method,
                "threshold": threshold,
                **result.summarize(threshold),
                "learned_kib": profile.learned_bytes(method) / 1024,
                "cnn_kib": WEIGHTS.stat().st_size / 1024 if method != "texture" else 0,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Fichier CSV à écrire")
    args = parser.parse_args()
    if not 0.65 <= args.threshold <= 1.5:
        parser.error("--threshold doit être compris entre 0.65 et 1.50")
    backbone = Backbone()
    catalog = read_catalog()
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    rows = evaluate(backbone, fit_demo(backbone, catalog), catalog, scenarios, args.threshold)
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    else:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
