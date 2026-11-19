"use strict";

const el = (id) => document.getElementById(id);
const state = {
  catalog: null, sample: null, file: null, profile: "demo",
  result: null, view: "heat", truth: false, request: null,
  benchmark: null, benchScenario: "normal",
};
const views = {
  heat: ["Carte d’anomalie", "Les couleurs chaudes indiquent un écart plus marqué."],
  boxes: ["Contours et boîtes", "Les boîtes entourent les régions au-dessus du seuil."],
  mask: ["Masque de prédiction", "Les pixels blancs sont signalés comme anormaux."],
};
const decimal = (value, digits = 2) =>
  Number(value).toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const percent = (value, digits = 1) => value == null ? "—" : decimal(value * 100, digits) + " %";
const threshold = () => Number(el("threshold").value);

function node(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== undefined) element.textContent = value;
  return element;
}

function button(className, label, action) {
  const control = node("button", className, label);
  control.type = "button";
  control.addEventListener("click", action);
  return control;
}

async function request(url, body, signal) {
  const response = await fetch(url, body ? { method: "POST", body, signal } : { signal });
  if (!response.ok) {
    let message = "La requête a échoué. Réessayez.";
    try {
      const { detail } = await response.json();
      if (typeof detail === "string") message = detail;
    } catch {
      // Préserve le message de repli si le serveur a renvoyé du texte.
    }
    throw new Error(message);
  }
  return response.json();
}

function notice(message, error = false) {
  el("notice").hidden = !message;
  el("notice").classList.toggle("error", error);
  el("notice").textContent = message;
}

function changePage(page) {
  for (const section of ["inspect", "evaluate"]) el(section).hidden = section !== page;
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    const active = tab.dataset.page === page;
    tab.classList.toggle("active", active);
    if (active) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);
}

function setProfile(id, train, calibration) {
  state.profile = id;
  el("profile-title").textContent = id === "demo" ? "Cuir · MVTec AD" : "Profil personnalisé";
  el("profile-note").textContent = `${train} apprentissage · ${calibration} calibration`;
  el("restore").hidden = id === "demo";
}

function restoreDemo() {
  setProfile("demo", state.catalog.train, state.catalog.calibration);
}

function showExamples() {
  const gallery = el("examples");
  for (const item of state.catalog.samples) {
    const tile = button("example", undefined, () => {
      state.sample = item.id;
      state.file = null;
      restoreDemo();
      analyze();
    });
    tile.dataset.sample = item.id;
    const preview = node("img");
    preview.src = "/api/sample/" + item.id;
    preview.alt = "";
    tile.append(preview, node("span", "", item.label));
    gallery.append(tile);
  }
}

function updateExamples() {
  document.querySelectorAll(".example").forEach((tile) => {
    tile.setAttribute("aria-pressed", String(tile.dataset.sample === state.sample && !state.file));
  });
}

function imageButton(url, label) {
  const control = button("image-button", undefined, () => {
    el("zoom-title").textContent = label;
    el("zoom-image").src = url;
    el("zoom-image").alt = label;
    el("zoom").showModal();
  });
  control.setAttribute("aria-label", "Agrandir · " + label);
  const image = node("img");
  image.src = url;
  image.alt = label;
  control.append(image);
  return control;
}

function card(title, subtitle, image, info) {
  const panel = node("article", "result-card");
  const head = node("div", "result-head");
  head.append(node("h3", "", title), node("p", "", subtitle));
  panel.append(head, image, info);
  return panel;
}

function download(url, name) {
  const anchor = node("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
}

function render() {
  const result = state.result;
  if (!result) return;
  const area = el("results");
  const scenario = el("scenario").selectedOptions[0].textContent;
  el("image-meta").textContent = `${result.name} · ${result.size.join(" × ")} px · ${scenario}`;
  const sourceInfo = node("div", "result-info");
  sourceInfo.append(node("span", "tag truth", result.label || "Image importée"));
  if (result.truth) {
    const toggle = button("text-button", state.truth ? "Masquer l’annotation" : "Afficher l’annotation", () => {
      state.truth = !state.truth;
      render();
      el("truth-toggle").focus();
    });
    toggle.id = "truth-toggle";
    toggle.setAttribute("aria-pressed", String(state.truth));
    sourceInfo.append(toggle);
  }
  const title = state.truth && result.truth ? "Annotation de référence" : "Image inspectée";
  area.replaceChildren(card("Image source", "Entrée commune", imageButton(
    state.truth && result.truth ? result.truth : result.source, title
  ), sourceInfo));

  for (const [method, row] of Object.entries(result.results)) {
    const [name, summary] = state.catalog.methods[method];
    const hot = row.regions > 0;
    const info = node("div", "result-info");
    info.append(node("span", hot ? "tag alert" : "tag", hot ? "Défaut suspecté" : "Conforme"));
    const metrics = node("dl", "metrics");
    const numbers = [
      ["Score", decimal(row.score) + "×"],
      ["Zones détectées", row.regions],
      ["Calcul CPU", decimal(row.ms, 1) + " ms"],
    ];
    if (row.iou !== undefined) numbers.push(["IoU", percent(row.iou)]);
    for (const [key, value] of numbers) {
      const line = node("div");
      line.append(node("dt", "", key), node("dd", "", value));
      metrics.append(line);
    }
    const exportButton = button("text-button", "Télécharger le masque", () => {
      const stem = result.name.replace(/\.[^.]*$/, "").replace(/[^\p{L}\p{N}_-]+/gu, "_");
      download(row.views.mask, `${stem}_${method}_masque.png`);
    });
    exportButton.setAttribute("aria-label", "Télécharger le masque · " + name);
    info.append(metrics, exportButton);
    area.append(card(name, summary, imageButton(row.views[state.view], `${name} · ${views[state.view][0]}`), info));
  }
}

async function analyze() {
  if (!state.catalog || (!state.sample && !state.file)) return;
  state.request?.abort();
  const controller = new AbortController();
  state.request = controller;
  state.truth = false;
  updateExamples();
  el("board").setAttribute("aria-busy", "true");
  notice("Inspection en cours…");
  const data = new FormData();
  data.set("scenario", el("scenario").value);
  data.set("threshold", threshold());
  data.set("profile", state.profile);
  if (state.file) data.set("image", state.file);
  else data.set("sample", state.sample);
  try {
    const result = await request("/api/analyze", data, controller.signal);
    if (state.request !== controller) return;
    state.result = result;
    notice(state.file && state.profile === "demo"
      ? "Comparaison avec le profil cuir. Pour un autre matériau, créez un profil avec vos images saines."
      : "");
    render();
  } catch (error) {
    if (state.request !== controller) return;
    state.result = null;
    el("results").replaceChildren();
    el("image-meta").textContent = "Image indisponible";
    notice(error.message, true);
  } finally {
    if (state.request === controller) el("board").setAttribute("aria-busy", "false");
  }
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  state.sample = null;
  analyze();
}

async function makeProfile(files) {
  if (!state.catalog || !files.length) return;
  if (files.length < 8 || files.length > 40) {
    notice("Sélectionnez entre 8 et 40 photos saines du même produit.", true);
    return;
  }
  state.request?.abort();
  const control = el("profile-action");
  control.disabled = true;
  notice("Création du profil et calibration des seuils…");
  const data = new FormData();
  for (const file of files) data.append("references", file);
  try {
    const profile = await request("/api/profile", data);
    setProfile(profile.id, profile.train, profile.calibration);
    if (state.file) analyze();
    else {
      state.sample = null;
      state.result = null;
      updateExamples();
      el("results").replaceChildren();
      el("image-meta").textContent = "Profil personnalisé actif";
      notice("Importez une image à inspecter avec ce profil.");
    }
  } catch (error) {
    notice(error.message, true);
  } finally {
    control.disabled = false;
  }
}

function renderBenchmark() {
  const rows = state.benchmark.filter((row) => row.scenario === state.benchScenario);
  const table = el("benchmark-body");
  table.replaceChildren();
  for (const row of rows) {
    const line = node("tr");
    const label = node("th", "", state.catalog.methods[row.method][0]);
    label.scope = "row";
    line.append(label);
    const data = [row.image_auc, row.pixel_auc, row.iou, row.precision, row.recall, row.fpr];
    line.append(...data.map((value) => node("td", "", percent(value))));
    line.append(node("td", "", decimal(row.median_ms, 1) + " ms"));
    table.append(line);
  }
  el("scenario-tabs").querySelectorAll("button").forEach((tab) => {
    const active = tab.dataset.scenario === state.benchScenario;
    tab.classList.toggle("selected", active);
    tab.setAttribute("aria-pressed", String(active));
  });
}

function updateThreshold() {
  el("threshold-value").textContent = decimal(threshold()) + "×";
  el("benchmark-threshold").textContent = decimal(threshold()) + "×";
}

async function runBenchmark() {
  if (!state.catalog) return;
  const control = el("run-benchmark");
  control.disabled = true;
  el("benchmark-results").hidden = true;
  el("download-csv").hidden = true;
  const cutoff = threshold();
  el("benchmark-state").textContent = "Calcul en cours sur 52 images et quatre scénarios…";
  const data = new FormData();
  data.set("threshold", cutoff);
  try {
    state.benchmark = await request("/api/evaluate", data);
    state.benchScenario = "normal";
    renderBenchmark();
    el("benchmark-results").hidden = false;
    el("download-csv").hidden = false;
    el("benchmark-state").textContent = "52 images par scénario · Seuil utilisé " + decimal(cutoff) + "×";
  } catch (error) {
    el("benchmark-state").textContent = error.message;
  } finally {
    control.disabled = false;
  }
}

function exportBenchmark() {
  const fields = Object.keys(state.benchmark[0]);
  const lines = state.benchmark.map((row) => fields.map((key) =>
    String(row[key] ?? "").replace(".", ",")).join(";"));
  const url = URL.createObjectURL(new Blob(["\ufeff", [fields.join(";"), ...lines].join("\r\n")], { type: "text/csv;charset=utf-8" }));
  download(url, "surfacelab_evaluation.csv");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function initialize() {
  try {
    state.catalog = await request("/api/catalog");
    restoreDemo();
    showExamples();
    for (const [key, label] of Object.entries(state.catalog.scenarios)) {
      const option = node("option", "", label);
      option.value = key;
      if (key === "normal") el("scenario").replaceChildren(option);
      else el("scenario").append(option);
      const tab = button("", label, () => {
        state.benchScenario = key;
        renderBenchmark();
      });
      tab.dataset.scenario = key;
      el("scenario-tabs").append(tab);
    }
    state.sample = state.catalog.samples[0].id;
    analyze();
  } catch (error) {
    notice(error.message, true);
    el("board").setAttribute("aria-busy", "false");
  }
}

document.querySelectorAll("[data-page]").forEach((tab) => tab.addEventListener("click", () => changePage(tab.dataset.page)));
document.querySelectorAll("[data-view]").forEach((tab) => tab.addEventListener("click", () => {
  state.view = tab.dataset.view;
  document.querySelectorAll("[data-view]").forEach((control) => {
    const active = control === tab;
    control.classList.toggle("selected", active);
    control.setAttribute("aria-pressed", String(active));
  });
  el("legend").hidden = state.view !== "heat";
  el("view-description").textContent = views[state.view][1];
  render();
}));

const upload = el("upload");
upload.addEventListener("click", () => el("image-file").click());
upload.addEventListener("dragover", (event) => { event.preventDefault(); upload.classList.add("drag"); });
upload.addEventListener("dragleave", () => upload.classList.remove("drag"));
upload.addEventListener("drop", (event) => {
  event.preventDefault();
  upload.classList.remove("drag");
  setFile(event.dataTransfer.files[0]);
});
el("image-file").addEventListener("change", (event) => {
  setFile(event.target.files[0]);
  event.target.value = "";
});
el("profile-action").addEventListener("click", () => el("profile-dialog").showModal());
el("choose-references").addEventListener("click", () => {
  el("profile-dialog").close();
  el("reference-files").click();
});
el("reference-files").addEventListener("change", (event) => {
  makeProfile(Array.from(event.target.files));
  event.target.value = "";
});
el("restore").addEventListener("click", () => {
  restoreDemo();
  if (!state.file) state.sample = state.catalog.samples[0].id;
  analyze();
});
el("scenario").addEventListener("change", analyze);
el("threshold").addEventListener("input", updateThreshold);
el("threshold").addEventListener("change", analyze);
el("run-benchmark").addEventListener("click", runBenchmark);
el("download-csv").addEventListener("click", exportBenchmark);
initialize();
