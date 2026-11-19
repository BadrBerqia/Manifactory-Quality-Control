"""API locale. Les profils restent en mémoire et les photos ne sont pas conservées."""

from collections import OrderedDict
from contextlib import asynccontextmanager
from hashlib import sha256
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inspection.benchmark import evaluate
from inspection.dataset import DATA, LABELS, fit_demo, read_catalog, read_sample
from inspection.media import MAX_UPLOAD, data_url, read_rgb, render_method, render_truth, truth_mask
from inspection.models import METHODS, ROOT, SCENARIOS, Backbone, fit, perturb, resize, score_all

WEB = ROOT / "web"
profiles = OrderedDict()
profile_lock = Lock()
benchmark_lock = Lock()


@asynccontextmanager
async def lifespan(application):
    application.state.catalog = read_catalog()
    application.state.examples = {row["id"]: row for row in application.state.catalog["test"]}
    application.state.backbone = Backbone()
    application.state.demo = fit_demo(application.state.backbone, application.state.catalog)
    yield


app = FastAPI(title="SurfaceLab", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/catalog")
def catalog():
    dataset = app.state.catalog
    # Un exemple par type dans la galerie, avec le jeu entier réservé à l'évaluation.
    picks = ["cut-000", "good-000", "color-000", "fold-000", "glue-000", "poke-000"]
    demos = []
    for key in picks:
        item = app.state.examples[key]
        demos.append({"id": key, "label": LABELS[item["kind"]], "kind": item["kind"]})
    return {
        "samples": demos,
        "methods": METHODS,
        "scenarios": SCENARIOS,
        "train": len(dataset["train"]),
        "calibration": len(dataset["calibration"]),
        "test": len(dataset["test"]),
    }


@app.get("/api/sample/{sample_id}")
def thumbnail(sample_id: str):
    item = app.state.examples.get(sample_id)
    if item is None:
        raise HTTPException(404, "Exemple introuvable.")
    return FileResponse(DATA / item["image"])


def read_upload(upload):
    content = upload.file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, f"Le fichier « {upload.filename} » dépasse 16 Mo.")
    try:
        return read_rgb(content)
    except ValueError as error:
        raise HTTPException(422, f"{upload.filename} · {error}") from None


def get_profile(profile_id):
    if profile_id == "demo":
        return app.state.demo
    with profile_lock:
        profile = profiles.get(profile_id)
        if profile is None:
            raise HTTPException(404, "Profil expiré. Rechargez les références ou sélectionnez le cuir MVTec.")
        profiles.move_to_end(profile_id)
        return profile


@app.post("/api/profile")
def make_profile(references: Annotated[list[UploadFile], File()]):
    if not 8 <= len(references) <= 40:
        raise HTTPException(422, "Sélectionnez entre 8 et 40 photos saines du même produit.")

    # Refuser les doubles exacts pour garder apprentissage et calibration disjoints.
    unique = set()
    images = []
    for upload in references:
        rgb, _ = read_upload(upload)
        digest = sha256(rgb.tobytes()).digest()
        if digest in unique:
            raise HTTPException(422, "Des références identiques ont été sélectionnées.")
        unique.add(digest)
        images.append(resize(rgb))
    cutoff = len(images) - max(2, len(images) // 4)
    with profile_lock:
        profile = fit(app.state.backbone, images[:cutoff], images[cutoff:])
        profile_id = uuid4().hex
        profiles[profile_id] = profile
        # Limite l'utilisation de la mémoire si plusieurs profils sont créés.
        if len(profiles) > 6:
            profiles.popitem(last=False)
    return {"id": profile_id, "train": cutoff, "calibration": len(images) - cutoff}


Threshold = Annotated[float, Form(ge=0.65, le=1.5)]


@app.post("/api/analyze")
def analyze(
    sample: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "demo",
    scenario: Annotated[str, Form()] = "normal",
    threshold: Threshold = 1.0,
    image: Annotated[UploadFile | None, File()] = None,
):
    if scenario not in SCENARIOS:
        raise HTTPException(422, "Scénario inconnu.")
    if image is not None:
        rgb, dimensions = read_upload(image)
        name = image.filename or "Image importée"
        label, truth = None, None
    else:
        item = app.state.examples.get(sample)
        if item is None:
            raise HTTPException(422, "Sélectionnez une image à inspecter.")
        rgb, dimensions = read_sample(item["image"])
        name = "cuir_" + item["id"] + ".jpg"
        label = LABELS[item["kind"]]
        truth = truth_mask(DATA / item["mask"]) if "mask" in item else None
    rgb = perturb(rgb, scenario)
    selected = get_profile(profile)
    results = {}
    for method, heat, elapsed in score_all(app.state.backbone, selected, rgb):
        result = render_method(rgb, heat, threshold)
        result["ms"] = elapsed
        if truth is not None:
            prediction = heat >= threshold
            result["iou"] = float((prediction & truth).sum() / (prediction | truth).sum())
        results[method] = result
    return {
        "name": name,
        "size": dimensions,
        "label": label,
        "source": data_url(rgb),
        "truth": render_truth(rgb, truth) if truth is not None else None,
        "threshold": threshold,
        "results": results,
    }


@app.post("/api/evaluate")
def benchmark(threshold: Threshold = 1.0):
    if not benchmark_lock.acquire(blocking=False):
        raise HTTPException(409, "Une évaluation est déjà en cours.")
    try:
        return evaluate(app.state.backbone, app.state.demo, app.state.catalog, SCENARIOS, threshold)
    finally:
        benchmark_lock.release()
