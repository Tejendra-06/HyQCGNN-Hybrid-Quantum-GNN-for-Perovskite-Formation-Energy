"use strict";

const EXAMPLES = ["SrTiO3", "BaTiO3", "LaAlO3", "KNbO3", "CaZrO3", "BiFeO3", "NdNiO3"];
const MODEL_ORDER = ["xgboost", "classical", "hybrid"];
const MODEL_SHORT = { xgboost: "XGBoost", classical: "Classical GNN", hybrid: "Hybrid quantum GNN" };
const MODEL_COLOR = { xgboost: "--accent-2", classical: "--accent", hybrid: "--accent-3" };
const MODEL_TEXT = {
  xgboost: "Gradient-boosted trees on the flattened graph (35 node + 40 edge descriptors). The production estimate.",
  classical: "One GENConv message-passing layer reduced to a 5×5 weight matrix and a linear read-out. Trained gradient-free.",
  hybrid: "GENConv output amplitude-encoded into 5 qubits, a trainable RX/RY/RZZ ansatz and learned read-out. Trained gradient-free.",
};
// diverging energy scale: stable (cyan) -> violet -> unstable (pink/amber)
const ENERGY_SCALE = [[0, "#06b6d4"], [0.3, "#6366f1"], [0.55, "#a855f7"], [0.8, "#ec4899"], [1, "#fbbf24"]];
const SIGNED_SCALE = [[0, "#06b6d4"], [0.5, "#8b5cf6"], [1, "#ec4899"]];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const state = { overview: null, result: null, batch: [], sort: { key: null, dir: 1 }, n: 1, viewer: null, hero: null };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ helpers
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const signed = (v, d = 3) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const formulaHTML = (f) => f.replace(/(\d+)/g, "<sub>$1</sub>");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeHTML = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
  return data;
}

function plotTheme(extra = {}) {
  const text = css("--text-2"), grid = css("--border");
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: text },
    margin: { l: 50, r: 16, t: 10, b: 40 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    hoverlabel: { bgcolor: css("--surface-solid"), bordercolor: css("--accent"), font: { family: "Inter, sans-serif", color: css("--text") } },
    transition: reduceMotion ? undefined : { duration: 600, easing: "cubic-in-out" },
    ...extra,
  };
}
const plotConfig = { displayModeBar: false, responsive: true };
const whenReady = (globalName) => new Promise((resolve) => {
  const check = () => (window[globalName] ? resolve(window[globalName]) : setTimeout(check, 50));
  check();
});

/** Animate a number into an element (easeOutExpo). */
function countUp(el, to, { digits = 0, format = (v) => v.toFixed(digits), duration = 1100 } = {}) {
  if (reduceMotion) { el.textContent = format(to); return; }
  const start = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - start) / duration);
    const e = k === 1 ? 1 : 1 - Math.pow(2, -10 * k);
    el.textContent = format(to * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ------------------------------------------------------------ motion helpers
const revealer = new IntersectionObserver((entries) => entries.forEach((e) => {
  if (e.isIntersecting) { e.target.classList.add("in"); revealer.unobserve(e.target); }
}), { threshold: 0.08 });

function reveal(root) {
  $$(".reveal", root).forEach((el, i) => {
    el.classList.remove("in");
    el.style.setProperty("--d", `${(i % 2) * 0.08 + Math.floor(i / 2) * 0.05}s`);
    revealer.observe(el);
  });
}

function bindCardFX() {
  document.addEventListener("pointermove", (e) => {
    const card = e.target.closest?.(".card");
    if (!card) return;
    const r = card.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    card.style.setProperty("--mx", `${x}px`);
    card.style.setProperty("--my", `${y}px`);
    if (card.hasAttribute("data-tilt") && !reduceMotion && card.classList.contains("in")) {
      const rx = (y / r.height - 0.5) * -6, ry = (x / r.width - 0.5) * 8;
      card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(0)`;
    }
  }, { passive: true });
  document.addEventListener("pointerout", (e) => {
    const card = e.target.closest?.("[data-tilt]");
    if (card && !card.contains(e.relatedTarget)) card.style.transform = "";
  });
}

// ------------------------------------------------------------------- tabs
function movePill() {
  const active = $(".tabs button[aria-selected='true']"), pill = $(".tab-pill");
  if (!active || !pill) return;
  pill.style.width = `${active.offsetWidth}px`;
  pill.style.transform = `translateX(${active.offsetLeft}px)`;
}

function showTab(name) {
  const current = $(".tabs button[aria-selected='true']")?.dataset.tab;
  if (current !== name) scrollTo({ top: 0 });
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".panel").forEach((p) => (p.hidden = p.dataset.panel !== name));
  movePill();
  if (name === "explore") renderHeatmap();
  if (name === "models") animateMeters();
  history.replaceState(null, "", name === "predict" ? location.pathname + location.search : `#${name}`);
  window.dispatchEvent(new Event("resize"));   // let Plotly re-measure freshly shown charts
}

// ---------------------------------------------------------------- predict
async function predict(formula) {
  formula = formula.trim();
  if (!formula) return;
  showTab("predict");
  $("#formula").value = formula;
  $("#form-error").hidden = true;
  $("#predict-btn").disabled = true;
  $("#loading").hidden = false;
  try {
    const [r] = await Promise.all([
      api(`/api/predict/${encodeURIComponent(formula)}`),
      new Promise((ok) => setTimeout(ok, reduceMotion ? 0 : 450)),   // let the loader be seen, not flash
    ]);
    state.result = r;
    $("#empty-state").hidden = true;
    $("#results").hidden = false;
    renderResult(r);
    reveal($("#results"));
    $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    const url = new URL(location.href);
    url.searchParams.set("f", r.formula);
    url.hash = "";
    history.replaceState(null, "", url);
  } catch (err) {
    $("#form-error").textContent = err.message;
    $("#form-error").hidden = false;
  } finally {
    $("#predict-btn").disabled = false;
    $("#loading").hidden = true;
  }
}

function renderResult(r, { animate = true } = {}) {
  const p = r.predictions, best = p.xgboost;
  $("#r-formula").innerHTML = formulaHTML(r.formula);
  if (!$(".card-hero .glow-orb")) $(".card-hero").insertAdjacentHTML("afterbegin", `<span class="glow-orb"></span>`);

  // badges + notices
  const badges = [], notices = [];
  if (r.structure_source === "dft") {
    badges.push(`<span class="badge good">DFT structure</span>`);
    badges.push(`<span class="badge ${r.split === "train" ? "warn" : "info"}">${r.split} split</span>`);
    if (r.split === "train") {
      notices.push(["info", `<strong>${formulaHTML(r.formula)} was in the training set.</strong> The models have seen its DFT value, so the agreement shown here is optimistic.`]);
    }
  } else {
    badges.push(`<span class="badge warn">Extrapolation</span>`);
    notices.push(["warn", `<strong>Not in the DFT data set.</strong> Predicted on an ideal cubic cell with a lattice constant estimated from Shannon radii (typical error ±${fmt(r.lattice_estimate_mae, 2)} Å).`]);
  }
  if (r.unknown_elements.length) {
    notices.push(["warn", `<strong>${r.unknown_elements.join(", ")} never appeared in training.</strong> The models only know 52 elements; treat this value as qualitative.`]);
  }
  $("#r-badges").innerHTML = badges.join("");
  $("#notices").innerHTML = notices.map(([k, t]) => `<div class="notice ${k}">${t}</div>`).join("");

  // headline
  if (animate) countUp($("#r-value"), best.e_form, { format: (v) => signed(v, 2) });
  else $("#r-value").textContent = signed(best.e_form, 2);
  $("#r-sub").innerHTML = `± ${fmt(best.typical_error, 2)} eV typical error · ${signed(best.e_form_per_atom, 3)} eV/atom · more stable than ${fmt(100 - r.percentile, 0)}% of known ABO<sub>3</sub>`;
  const kv = [];
  if (r.dft !== null) {
    const err = best.e_form - r.dft, cls = Math.abs(err) < 0.1 ? "delta-good" : Math.abs(err) < 0.3 ? "delta-warn" : "delta-bad";
    kv.push(["DFT reference", `${signed(r.dft, 2)} eV`], ["XGBoost − DFT", `<span class="${cls}">${signed(err, 3)} eV</span>`]);
  } else {
    kv.push(["DFT reference", "not available"], ["Cell", "ideal cubic"]);
  }
  kv.push(["Classical GNN", `${signed(p.classical.e_form, 2)} eV`], ["Hybrid quantum GNN", `${signed(p.hybrid.e_form, 2)} eV`]);
  $("#r-kv").innerHTML = kv.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  renderModelChart(r);
  renderStability(r.descriptors);
  renderStructure(r);
  renderShap(r.explanation);
  renderQuantum(r.quantum);
  renderHistogram(r);
  renderNeighbours(r.neighbours);
}

async function renderModelChart(r) {
  const Plotly = await whenReady("Plotly");
  const names = MODEL_ORDER.map((k) => MODEL_SHORT[k]);
  const vals = MODEL_ORDER.map((k) => r.predictions[k].e_form);
  const errs = MODEL_ORDER.map((k) => r.predictions[k].typical_error);
  const colors = MODEL_ORDER.map((k) => css(MODEL_COLOR[k]));
  const shapes = [], annotations = [];
  if (r.dft !== null) {
    shapes.push({ type: "line", x0: r.dft, x1: r.dft, yref: "paper", y0: 0, y1: 1, line: { color: css("--text"), width: 2, dash: "dot" } });
    annotations.push({ x: r.dft, y: 1, yref: "paper", yanchor: "bottom", text: `DFT ${signed(r.dft, 2)}`, showarrow: false, font: { color: css("--text"), size: 11 } });
  }
  Plotly.react("chart-models", [{
    type: "bar", orientation: "h", y: names, x: vals,
    marker: { color: colors, line: { color: colors, width: 1 }, opacity: 0.92 },
    error_x: { type: "data", array: errs, color: css("--text-2"), thickness: 1.4, width: 6 },
    hovertemplate: "%{y}: %{x:.3f} eV<extra></extra>",
  }], plotTheme({
    margin: { l: 130, r: 20, t: r.dft !== null ? 26 : 10, b: 44 }, bargap: 0.38,
    xaxis: { title: { text: "Formation energy (eV / cell)" }, gridcolor: css("--border"), zerolinecolor: css("--border-strong") },
    yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" },
    shapes, annotations,
  }), plotConfig);
}

function gauge({ label, value, min, max, lo, hi, verdict, good, digits = 3 }) {
  if (value === null || value === undefined) {
    return `<div class="gauge"><div class="gauge-top"><span class="gauge-label">${label}</span><span class="gauge-value">—</span></div>
      <div class="gauge-verdict muted">${verdict || "not defined for this compound"}</div></div>`;
  }
  const pct = (v) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
  return `<div class="gauge">
    <div class="gauge-top"><span class="gauge-label">${label}</span><span class="gauge-value">${fmt(value, digits)}</span></div>
    <div class="gauge-track"><div class="gauge-zone" style="left:${pct(lo)}%;width:${pct(hi) - pct(lo)}%"></div>
      <div class="gauge-marker" data-left="${pct(value)}"></div></div>
    <div class="gauge-scale"><span>${min}</span><span>stable ${lo}–${hi}</span><span>${max}</span></div>
    <div class="gauge-verdict ${good ? "delta-good" : "delta-warn"}">${good ? "✓ " : "! "}${verdict}</div></div>`;
}

function renderStability(d) {
  const t = d.tolerance_factor, mu = d.octahedral_factor;
  const tGood = t !== null && t >= 0.825 && t <= 1.059 && mu >= 0.414;
  const ox = d.oxidation_states;
  $("#stability").innerHTML =
    gauge({ label: "Goldschmidt tolerance factor t", value: t, min: 0.6, max: 1.3, lo: 0.825, hi: 1.059, verdict: d.goldschmidt_verdict, good: tGood }) +
    gauge({ label: "Bartel τ factor", value: d.tau, min: 0, max: 12, lo: 0, hi: 4.18, verdict: d.tau_verdict || "defined only when r(A) > r(B)", good: d.tau !== null && d.tau < 4.18, digits: 2 }) +
    `<div class="facts">
      <div class="fact"><div class="l">Oxidation states</div><div class="v">${ox.A > 0 ? "+" : ""}${ox.A} / +${ox.B} / −2</div></div>
      <div class="fact"><div class="l">Octahedral factor μ</div><div class="v">${fmt(mu, 3)}</div></div>
      <div class="fact"><div class="l">Lattice a</div><div class="v">${fmt(d.lattice_constant, 3)} Å</div></div>
      <div class="fact"><div class="l">r(A) XII / r(B) VI</div><div class="v">${fmt(d.radius_a, 2)} / ${fmt(d.radius_b, 2)} Å</div></div>
      <div class="fact"><div class="l">A–O / B–O bond</div><div class="v">${fmt(d.bond_a_o, 2)} / ${fmt(d.bond_b_o, 2)} Å</div></div>
      <div class="fact"><div class="l">Density</div><div class="v">${fmt(d.density, 2)} g/cm³</div></div>
    </div>`;
  // slide the markers in from the left once laid out
  requestAnimationFrame(() => requestAnimationFrame(() =>
    $$("#stability .gauge-marker").forEach((m) => (m.style.left = `${m.dataset.left}%`))));
}

// ------------------------------------------------------- 3D crystal view
const sceneReady = () => new Promise((resolve) => {
  if (window.PerovskiteScene) resolve(window.PerovskiteScene);
  else addEventListener("scene-ready", () => resolve(window.PerovskiteScene), { once: true });
});

async function renderStructure(r) {
  const { CrystalScene } = await sceneReady();
  if (!state.viewer) state.viewer = new CrystalScene($("#viewer"));
  const kinds = ["A", "B", "O", "O", "O"];
  state.viewer.setStructure({
    sites: r.structure.sites.map((s, i) => ({ kind: kinds[i], frac: s.frac })),
    a: r.descriptors.lattice_constant, n: state.n, bond: r.descriptors.bond_b_o,
  });
  $("#viewer-legend").innerHTML = [["dot-a", `${r.a} · A`], ["dot-b", `${r.b} · B`], ["dot-o", "O"]]
    .map(([c, t]) => `<span><i class="dot ${c}"></i>${t}</span>`).join("");
}

async function initHero() {
  const { CrystalScene, idealSites } = await sceneReady();
  try {
    state.hero = new CrystalScene($("#hero-stage"), { hero: true });
    state.hero.setStructure({ sites: idealSites, a: 3.9, n: 3, bond: 1.95 });
  } catch (err) {
    console.warn("3D hero disabled:", err);   // e.g. no WebGL: the page still works
  }
}

async function renderShap(items) {
  const Plotly = await whenReady("Plotly");
  const rows = [...items].reverse();
  Plotly.react("chart-shap", [{
    type: "bar", orientation: "h", y: rows.map((d) => d.feature), x: rows.map((d) => d.contribution),
    marker: { color: rows.map((d) => (d.contribution < 0 ? css("--accent-2") : css("--accent-3"))), opacity: 0.9 },
    hovertemplate: "%{y}<br>%{x:+.3f} eV<extra></extra>",
  }], plotTheme({
    margin: { l: 220, r: 16, t: 6, b: 44 }, bargap: 0.3,
    xaxis: { title: { text: "← lowers energy · raises energy →" }, gridcolor: css("--border"), zerolinecolor: css("--border-strong") },
    yaxis: { gridcolor: "rgba(0,0,0,0)", automargin: true },
  }), plotConfig);
}

async function renderQuantum(q) {
  const Plotly = await whenReady("Plotly");
  const labels = ["A", "B", "O₁", "O₂", "O₃"];
  const m = Math.max(...q.matrix.flat().map(Math.abs));
  Plotly.react("chart-matrix", [{
    type: "heatmap", z: q.matrix, x: labels, y: labels, colorscale: SIGNED_SCALE, zmin: -m, zmax: m,
    showscale: false, xgap: 3, ygap: 3, hovertemplate: "%{y}–%{x}: %{z:.3f}<extra></extra>",
  }], plotTheme({ margin: { l: 30, r: 6, t: 4, b: 26 }, yaxis: { autorange: "reversed" }, xaxis: {}, transition: undefined }), plotConfig);
  Plotly.react("chart-z", [{
    type: "bar", x: q.expectations.map((_, i) => `q${i}`), y: q.expectations,
    marker: { color: q.expectations.map((v) => (v < 0 ? css("--accent-2") : css("--accent-3"))), opacity: 0.92 },
    hovertemplate: "⟨Z⟩ %{x}: %{y:.3f}<extra></extra>",
  }], plotTheme({ margin: { l: 36, r: 6, t: 4, b: 26 }, bargap: 0.35, yaxis: { range: [-1, 1], gridcolor: css("--border"), zerolinecolor: css("--border-strong") } }), plotConfig);
}

async function renderHistogram(r) {
  const Plotly = await whenReady("Plotly");
  const h = state.overview.histogram;
  const centers = h.counts.map((_, i) => (h.edges[i] + h.edges[i + 1]) / 2);
  const pred = r.predictions.xgboost.e_form;
  const shapes = [], annotations = [];
  const mark = (x, color, label, dash) => {
    shapes.push({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color, width: 2.5, dash } });
    if (label) annotations.push({ x, y: 1, yref: "paper", yanchor: "bottom", text: label, showarrow: false, font: { color, size: 11 } });
  };
  mark(pred, css("--accent-3"), "predicted");
  if (r.dft !== null) mark(r.dft, css("--text"), "", "dot");
  // colour every bin by energy so the plot reads as the same scale as the explorer
  const lo = h.edges[0], hi = h.edges[h.edges.length - 1];
  Plotly.react("chart-hist", [{
    type: "bar", x: centers, y: h.counts, width: (h.edges[1] - h.edges[0]) * 0.86,
    marker: { color: centers, colorscale: ENERGY_SCALE, cmin: lo, cmax: hi, opacity: 0.75 },
    hovertemplate: "%{x:.2f} eV: %{y} compounds<extra></extra>",
  }], plotTheme({
    margin: { l: 48, r: 12, t: 22, b: 44 }, shapes, annotations,
    xaxis: { title: { text: "Formation energy (eV / cell)" }, gridcolor: "rgba(0,0,0,0)" },
    yaxis: { title: { text: "Compounds" }, gridcolor: css("--border") },
  }), plotConfig);
}

function renderNeighbours(list) {
  const max = Math.max(...list.map((n) => n.distance), 1e-6);
  $("#neighbours").innerHTML = list.map((n, i) => `
    <li data-f="${n.formula}" title="Predict ${n.formula}" style="--i:${i}">
      <span class="f">${formulaHTML(n.formula)}</span>
      <span class="bar" title="similarity"><i style="width:${100 - (n.distance / max) * 70}%"></i></span>
      <span class="e">${signed(n.dft, 2)} eV</span>
    </li>`).join("");
}

// ------------------------------------------------------------------ batch
function parseBatch(text) {
  return text.split(/[\n,;\t]+/).map((s) => s.trim().replace(/^"|"$/g, ""))
    .filter((s) => s && !/^formula$/i.test(s));
}

async function runBatch() {
  const formulas = parseBatch($("#batch-input").value);
  $("#batch-error").hidden = true;
  if (!formulas.length) return;
  $("#batch-run").disabled = true;
  $("#loading").hidden = false;
  try {
    const { rows } = await api("/api/batch", { formulas });
    state.batch = rows;
    state.sort = { key: null, dir: 1 };
    renderBatch();
  } catch (err) {
    $("#batch-error").textContent = err.message;
    $("#batch-error").hidden = false;
  } finally {
    $("#batch-run").disabled = false;
    $("#loading").hidden = true;
  }
}

function renderBatch() {
  const rows = [...state.batch];
  const { key, dir } = state.sort;
  if (key) {
    rows.sort((a, b) => {
      const x = a[key] ?? (dir > 0 ? Infinity : -Infinity), y = b[key] ?? (dir > 0 ? Infinity : -Infinity);
      return (x > y ? 1 : x < y ? -1 : 0) * dir;
    });
  }
  $$("#batch-table th").forEach((th) => (th.textContent = th.textContent.replace(/ [▲▼]$/, "") + (th.dataset.key === key ? (dir > 0 ? " ▲" : " ▼") : "")));
  const ok = state.batch.filter((r) => !r.error).length;
  $("#batch-count").textContent = `${ok} predicted${state.batch.length > ok ? `, ${state.batch.length - ok} invalid` : ""}`;
  $("#batch-download").disabled = !ok;
  $("#batch-table tbody").innerHTML = rows.map((r, i) => r.error
    ? `<tr class="err" style="animation-delay:${Math.min(i, 30) * 20}ms"><td>${escapeHTML(r.input)}</td><td colspan="7">${escapeHTML(r.error)}</td></tr>`
    : `<tr style="animation-delay:${Math.min(i, 30) * 20}ms"><td class="f" data-f="${r.formula}">${formulaHTML(r.formula)}</td>
        <td class="num"><strong>${fmt(r.xgboost)}</strong></td><td class="num">${fmt(r.classical)}</td><td class="num">${fmt(r.hybrid)}</td>
        <td class="num">${fmt(r.dft, 2)}</td><td class="num">${fmt(r.tolerance_factor)}</td><td class="num">${fmt(r.tau, 2)}</td>
        <td>${r.structure_source === "dft" ? `<span class="badge good">DFT</span>` : `<span class="badge warn">estimated</span>`}</td></tr>`).join("")
    || `<tr><td colspan="8" class="muted center">No results.</td></tr>`;
}

function downloadCSV() {
  const cols = ["input", "formula", "xgboost", "classical", "hybrid", "dft", "split", "structure_source",
    "tolerance_factor", "octahedral_factor", "tau", "lattice_constant", "error"];
  const cell = (v) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  const csv = [cols.join(","), ...state.batch.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "perovskite_predictions.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- explore
let heatmapDrawn = false;
async function renderHeatmap(force = false) {
  if (!state.overview || (heatmapDrawn && !force)) return;
  const Plotly = await whenReady("Plotly");
  const els = state.overview.elements;
  const pos = Object.fromEntries(els.map((e, i) => [e, i]));
  const z = els.map(() => els.map(() => null));
  state.overview.heatmap.forEach(({ a, b, e }) => (z[pos[a]][pos[b]] = e));
  Plotly.react("chart-heatmap", [{
    type: "heatmap", z, x: els, y: els, colorscale: ENERGY_SCALE,
    colorbar: { title: { text: "eV / cell", side: "right" }, thickness: 12, outlinewidth: 0 },
    hovertemplate: "<b>%{y}%{x}O₃</b><br>DFT %{z:.2f} eV<extra></extra>", xgap: 1, ygap: 1,
  }], plotTheme({
    margin: { l: 56, r: 10, t: 10, b: 56 }, transition: undefined,
    xaxis: { title: { text: "B site" }, tickfont: { size: 9 }, dtick: 1, tickangle: 0, gridcolor: "rgba(0,0,0,0)", side: "bottom" },
    yaxis: { title: { text: "A site" }, tickfont: { size: 9 }, dtick: 1, gridcolor: "rgba(0,0,0,0)", autorange: "reversed" },
  }), plotConfig);
  if (!heatmapDrawn) {
    $("#chart-heatmap").on("plotly_click", (ev) => {
      const pt = ev.points[0];
      if (pt.z !== null) predict(`${pt.y}${pt.x}O3`);
    });
  }
  heatmapDrawn = true;
}

// ----------------------------------------------------------------- models
function renderModels(o) {
  $("#model-cards").innerHTML = MODEL_ORDER.map((k) => {
    const m = o.models[k];
    return `<article class="card span-4 model-card" data-tilt>
      <div class="card-head"><h3>${m.label}</h3>${k === "xgboost" ? `<span class="badge good">best</span>` : `<span class="badge neutral">research</span>`}</div>
      <div class="score">${fmt(m.test_r2, 3)}<small>test R²</small></div>
      <div class="meter"><i data-w="${Math.max(0, m.test_r2) * 100}"></i></div>
      <dl><dt>Test MAE</dt><dd>${fmt(m.test_mae, 3)} eV / cell</dd>
          <dt>Training set</dt><dd>${m.n_train ? m.n_train.toLocaleString() : "—"} compounds</dd>
          ${m.n_parameters ? `<dt>Parameters</dt><dd>${m.n_parameters}</dd>` : ""}</dl>
      <p>${MODEL_TEXT[k]}</p></article>`;
  }).join("");
  $$("#model-cards .card").forEach((c) => c.classList.add("in"));
}

function animateMeters() {
  $$("#model-cards .meter i").forEach((m) => { m.style.width = "0"; });
  requestAnimationFrame(() => requestAnimationFrame(() =>
    $$("#model-cards .meter i").forEach((m) => (m.style.width = `${m.dataset.w}%`))));
}

// ------------------------------------------------------------------ theme
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch { /* storage blocked */ }
  $('meta[name="theme-color"]').content = css("--bg");
  state.hero?.refreshTheme();
  if (state.result) renderResult(state.result, { animate: false });
  if (heatmapDrawn) renderHeatmap(true);
}

// ------------------------------------------------------------------- init
async function init() {
  $("#examples").insertAdjacentHTML("beforeend",
    EXAMPLES.map((f) => `<button type="button" class="chip" data-f="${f}">${formulaHTML(f)}</button>`).join(""));
  document.addEventListener("click", (e) => {
    const f = e.target.closest("[data-f]");
    if (f) predict(f.dataset.f);
  });
  $("#predict-form").addEventListener("submit", (e) => { e.preventDefault(); predict($("#formula").value); });
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  addEventListener("resize", movePill);
  document.fonts?.ready.then(movePill);
  $$("#view-seg button").forEach((b) => b.addEventListener("click", () => {
    state.n = Number(b.dataset.n);
    $$("#view-seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.viewer?.setOptions({ n: state.n });
  }));
  for (const [id, key] of [["#toggle-poly", "polyhedra"], ["#toggle-spin", "autoRotate"]]) {
    $(id).addEventListener("click", (e) => {
      const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
      e.currentTarget.setAttribute("aria-pressed", String(on));
      state.viewer?.setOptions({ [key]: on });
    });
  }
  $("#theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    applyTheme(current === "dark" ? "light" : "dark");
  });
  $("#batch-run").addEventListener("click", runBatch);
  $("#batch-download").addEventListener("click", downloadCSV);
  $("#batch-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) $("#batch-input").value = parseBatch(await file.text()).join("\n");
    e.target.value = "";
  });
  $$("#batch-table th").forEach((th) => th.addEventListener("click", () => {
    const key = th.dataset.key;
    state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
    renderBatch();
  }));
  window.addEventListener("hashchange", () => {
    const t = location.hash.slice(1);
    if (["predict", "batch", "explore", "models"].includes(t)) showTab(t);
  });
  bindCardFX();
  movePill();
  initHero();

  const status = $("#status-dot");
  try {
    state.overview = await api("/api/overview");
    status.className = "status ok";
    status.lastElementChild.textContent = `${state.overview.n_compounds.toLocaleString()} compounds · ${state.overview.elements.length} elements · models ready`;
    $('[data-count="2652"]').dataset.count = state.overview.n_compounds;
    $('[data-count="52"]').dataset.count = state.overview.elements.length;
    const best = state.overview.models.xgboost.test_r2;
    if (best) $('[data-count="0.96"]').dataset.count = best.toFixed(2);
    renderModels(state.overview);
  } catch {
    status.className = "status down";
    status.lastElementChild.textContent = "Server unavailable";
  }
  $$("#stats [data-count]").forEach((el) => {
    const digits = Number(el.dataset.digits || 0);
    countUp(el, Number(el.dataset.count), { digits, format: (v) => (digits ? v.toFixed(digits) : Math.round(v).toLocaleString()), duration: 1600 });
  });
  if (!state.overview) return;

  const params = new URLSearchParams(location.search);
  const tab = location.hash.slice(1);
  if (params.get("f")) predict(params.get("f"));
  else if (["batch", "explore", "models"].includes(tab)) showTab(tab);
}

init();
