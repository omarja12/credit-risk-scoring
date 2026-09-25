import { scoreApplicant } from "./score.js";

const NEVER = 999;

const DEFAULTS = {
  annual_income: 62000,
  employment_length_years: 5,
  dti: 18,
  credit_utilization: 28,
  num_late_payments_2y: 0,
  months_since_last_delinquency: NEVER,
  has_bankruptcy_10y: 0,
  loan_purpose: "debt_consolidation",
};

const PRESETS = {
  strong: { annual_income: 95000, employment_length_years: 12, dti: 9, credit_utilization: 12,
    num_late_payments_2y: 0, months_since_last_delinquency: NEVER, has_bankruptcy_10y: 0,
    loan_purpose: "home_improvement" },
  borderline: { annual_income: 42000, employment_length_years: 2, dti: 31, credit_utilization: 58,
    num_late_payments_2y: 1, months_since_last_delinquency: 20, has_bankruptcy_10y: 0,
    loan_purpose: "debt_consolidation" },
  risky: { annual_income: 36000, employment_length_years: 1, dti: 38, credit_utilization: 78,
    num_late_payments_2y: 2, months_since_last_delinquency: 8, has_bankruptcy_10y: 0,
    loan_purpose: "small_business" },
};

const PURPOSES = {
  debt_consolidation: "Debt consolidation", credit_card: "Credit card refinancing",
  home_improvement: "Home improvement", car: "Car", medical: "Medical", small_business: "Small business",
};

const OUTCOME_SUB = {
  Approve: "Within lending policy",
  Refer: "Send to an underwriter",
  Decline: "Outside lending policy",
};



const $ = (sel) => document.querySelector(sel);
const money = (v) => "$" + Math.round(v).toLocaleString("en-US");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function formatValue(key, v) {
  switch (key) {
    case "annual_income": return money(v);
    case "employment_length_years": return plural(v, "year");
    case "dti":
    case "credit_utilization": return `${v}%`;
    case "num_late_payments_2y": return ["None", "One", "Two"][v] ?? "Three or more";
    case "months_since_last_delinquency": return v >= NEVER ? "Never" : `${plural(v, "month")} ago`;
    case "has_bankruptcy_10y": return v ? "Yes" : "No";
    case "loan_purpose": return PURPOSES[v] ?? v;
    default: return String(v);
  }
}

// ------------------------------------------------------------------ state
let card;
let state = { ...DEFAULTS };
let lastMonths = 12;

function readUrl() {
  const p = new URLSearchParams(location.search);
  const s = { ...DEFAULTS };
  for (const key of Object.keys(DEFAULTS)) {
    if (!p.has(key)) continue;
    const raw = p.get(key);
    if (key === "loan_purpose") { if (raw in PURPOSES) s[key] = raw; continue; }
    const n = Number(raw);
    if (Number.isFinite(n)) s[key] = n;
  }
  return s;
}

let urlTimer;
function writeUrl() {
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(state)) if (v !== DEFAULTS[k]) p.set(k, v);
    const q = p.toString();
    history.replaceState(null, "", q ? `?${q}` : location.pathname);
  }, 150);
}

// ------------------------------------------------------------------ form
const form = $("#applicant");
const ranges = [...form.querySelectorAll('input[type="range"]')];

function setRangeFill(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--fill", `${pct}%`);
}

function syncForm() {
  for (const input of ranges) {
    const key = input.name;
    const value = key === "months_since_last_delinquency"
      ? (state[key] >= NEVER ? lastMonths : state[key]) : state[key];
    input.value = value;
    setRangeFill(input);
    form.querySelector(`[data-out="${key}"]`).textContent = formatValue(key, Number(input.value));
  }
  const delinquent = state.months_since_last_delinquency < NEVER;
  $("#months-field").hidden = !delinquent;
  setSegment("ever_delinquent", delinquent ? 1 : 0);
  setSegment("num_late_payments_2y", Math.min(state.num_late_payments_2y, 3));
  setSegment("has_bankruptcy_10y", state.has_bankruptcy_10y);
  $("#loan_purpose").value = state.loan_purpose;
}

function setSegment(name, value) {
  form.querySelectorAll(`[data-name="${name}"] button`).forEach((b) => {
    const on = Number(b.dataset.value) === Number(value);
    b.setAttribute("aria-checked", on);
    b.tabIndex = on ? 0 : -1;
  });
}

function update(changes, { fromPreset = null } = {}) {
  state = { ...state, ...changes };
  if (state.months_since_last_delinquency < NEVER) lastMonths = state.months_since_last_delinquency;
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.setAttribute("aria-pressed", b.dataset.preset === fromPreset));
  syncForm();
  render();
  writeUrl();
}

ranges.forEach((input) =>
  input.addEventListener("input", () => update({ [input.name]: Number(input.value) })));

$("#loan_purpose").addEventListener("change", (e) => update({ loan_purpose: e.target.value }));

form.querySelectorAll(".segmented").forEach((group) => {
  const name = group.dataset.name;
  const choose = (btn) => {
    const v = Number(btn.dataset.value);
    if (name === "ever_delinquent") {
      update({ months_since_last_delinquency: v ? lastMonths : NEVER });
    } else {
      update({ [name]: v });
    }
  };
  group.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) choose(b); });
  group.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const buttons = [...group.querySelectorAll("button")];
    const i = buttons.findIndex((b) => b.getAttribute("aria-checked") === "true");
    const step = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
    const next = buttons[(i + step + buttons.length) % buttons.length];
    choose(next);
    next.focus();
  });
});

document.querySelectorAll("[data-preset]").forEach((b) =>
  b.addEventListener("click", () => update({ ...PRESETS[b.dataset.preset] }, { fromPreset: b.dataset.preset })));

$("#copy-link").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  try {
    await navigator.clipboard.writeText(location.href);
    btn.textContent = "Link copied";
  } catch {
    btn.textContent = "Copy the address bar to share";
  }
  setTimeout(() => { btn.textContent = "Copy link to this applicant"; }, 2200);
});

// ------------------------------------------------------------------ decision
const pctOf = (s) => {
  const { score_min, score_max } = card.meta;
  return ((Math.min(Math.max(s, score_min), score_max) - score_min) / (score_max - score_min)) * 100;
};

function buildBand() {
  const { score_max } = card.meta;
  const track = $("#band-track");
  // grades are ordered best first; draw from lowest score to highest
  const zones = card.grades.map((g, i) => ({
    grade: g.grade, lo: g.min_score, hi: i === 0 ? score_max : card.grades[i - 1].min_score,
  })).reverse();
  track.innerHTML = zones.map((z) =>
    `<div class="band-zone" data-grade="${z.grade}" style="flex:${z.hi - z.lo} 0 0"><b style="left:${pctOf((z.lo + z.hi) / 2)}%">${z.grade}</b></div>`
  ).join("");
}

function render() {
  const r = scoreApplicant(card, state);

  const sheet = $("#decision");
  sheet.dataset.outcome = r.decision;
  $("#score").textContent = r.score;
  $("#outcome").textContent = r.decision;
  $("#outcome-sub").textContent = OUTCOME_SUB[r.decision];
  $("#pd").textContent = `${(r.pd * 100).toFixed(1)}%`;
  $("#grade").innerHTML = `${r.grade}<small>${r.gradeLabel}</small>`;

  document.querySelectorAll(".band-zone").forEach((z) =>
    z.classList.toggle("is-current", z.dataset.grade === r.grade));
  $("#band-marker").style.left = `${pctOf(r.score)}%`;

  // reason codes: the factors losing the most points against their best range
  const reasons = r.factors.filter((f) => f.lost >= 3).sort((a, b) => b.lost - a.lost).slice(0, 4);
  const worst = reasons.length ? reasons[0].lost : 1;
  $("#reasons").innerHTML = reasons.length
    ? reasons.map((f) => `
      <li>
        <span class="reason-name">${f.feature.label}</span>
        <span class="reason-lost">−${Math.round(f.lost)} pts</span>
        <span class="reason-detail">This applicant: ${formatValue(f.feature.key, state[f.feature.key])}.
          Full points: ${f.feature.bins[f.feature.best].text.toLowerCase()}.</span>
        <span class="reason-bar"><i style="width:${(f.lost / worst) * 100}%"></i></span>
      </li>`).join("")
    : `<li class="clean">Every factor is in its best range.</li>`;

  // scorecard grid highlights
  r.factors.forEach((f) => {
    document.querySelectorAll(`[data-row="${f.feature.key}"] .cell`).forEach((c, i) =>
      c.classList.toggle("is-current", i === f.bin));
  });
  $("#grid-total").innerHTML =
    `Total for this applicant: <strong>${Math.round(r.raw)} points</strong>, a score of <strong>${r.score}</strong>`;

  $("#dock-score").textContent = r.score;
  const dockOutcome = $("#dock-outcome");
  dockOutcome.textContent = r.decision;
  dockOutcome.style.color = "var(--state-on-dark)";
  $("#dock-grade").textContent = `Grade ${r.grade}`;
}

// ------------------------------------------------------------------ scorecard grid
function buildGrid() {
  const maxIv = Math.max(...card.features.map((f) => f.iv));
  $("#grid").innerHTML = card.features.map((f) => `
    <div class="grid-row" data-row="${f.key}">
      <div class="grid-name">
        <strong>${f.label}</strong>
        <span>Information value ${f.iv.toFixed(3)}</span>
        <span class="grid-iv" aria-hidden="true"><i style="width:${(f.iv / maxIv) * 100}%"></i></span>
      </div>
      <div class="grid-cells">
        ${f.bins.map((b) => `
          <div class="cell">
            <span class="cell-range">${b.text}</span>
            <span class="cell-pts">${Math.round(b.points)}</span>
          </div>`).join("")}
      </div>
    </div>`).join("");
}

// ------------------------------------------------------------------ performance
function buildPerformance() {
  const m = card.metrics;
  $("#perf-intro").textContent =
    `Measured on ${m.n_test.toLocaleString("en-US")} applicants the model never saw during training, ` +
    `${(m.default_rate * 100).toFixed(1)}% of whom defaulted. Typical application scorecards reach ` +
    `an AUC between 0.65 and 0.75.`;

  $("#intro-stats").innerHTML = [
    ["Applicants scored in training", (m.n_train + m.n_test).toLocaleString("en-US")],
    ["Factors in the scorecard", String(card.features.length)],
    ["AUC on held-out applicants", m.test_auc.toFixed(3)],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  $("#metrics").innerHTML = [
    [m.test_auc.toFixed(3), "AUC", "The chance a borrower who defaulted scores lower than one who repaid. 0.5 is a coin toss."],
    [m.test_gini.toFixed(3), "Gini", "The same measure on a 0 to 1 scale, as risk teams usually report it."],
    [m.test_ks.toFixed(3), "KS", "The largest gap between the score distributions of good and bad loans."],
  ].map(([v, name, text]) => `<div class="metric"><strong>${v}</strong><span>${name}</span><p>${text}</p></div>`).join("");

  // calibration: predicted vs actual default rate per grade
  const cal = card.calibration;
  const W = 360, H = 220, L = 34, B = 26, T = 8;
  const top = Math.ceil(Math.max(...cal.flatMap((c) => [c.predicted, c.actual])) * 100 / 4) * 4 / 100;
  const y = (v) => T + (H - T - B) * (1 - v / top);
  const slot = (W - L) / cal.length;
  const bw = Math.min(18, slot / 3.2);
  const ticks = [0, top / 2, top];
  $("#calibration").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted and actual default rates by grade closely match">
      ${ticks.map((t) => `<line x1="${L}" x2="${W}" y1="${y(t)}" y2="${y(t)}" stroke="#EFEFEC"/>
        <text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${(t * 100).toFixed(0)}%</text>`).join("")}
      ${cal.map((c, i) => {
        const cx = L + slot * i + slot / 2;
        return `
          <rect x="${cx - bw - 1.5}" y="${y(c.predicted)}" width="${bw}" height="${y(0) - y(c.predicted)}" fill="#D8D8D4" rx="1.5"/>
          <rect x="${cx + 1.5}" y="${y(c.actual)}" width="${bw}" height="${y(0) - y(c.actual)}" fill="#1B1B1F" rx="1.5"/>
          <text x="${cx}" y="${H - 8}" text-anchor="middle" style="fill:#1B1B1F;font-weight:600">${c.grade}</text>`;
      }).join("")}
    </svg>
    <div class="legend"><span><i style="background:#D8D8D4"></i>Predicted</span><span><i style="background:#1B1B1F"></i>Actual</span></div>`;

  // ROC curve
  const S = 220, P = 30;
  const sx = (v) => P + v * (S - P - 6), sy = (v) => (S - P) - v * (S - P - 6);
  const pts = card.roc.map(([f, t]) => `${sx(f).toFixed(1)},${sy(t).toFixed(1)}`).join(" ");
  $("#roc").innerHTML = `
    <svg viewBox="0 0 ${S} ${S}" role="img" aria-label="ROC curve with AUC ${m.test_auc.toFixed(2)}">
      <rect x="${sx(0)}" y="${sy(1)}" width="${sx(1) - sx(0)}" height="${sy(0) - sy(1)}" fill="none" stroke="#EFEFEC"/>
      <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#C9C9C4" stroke-dasharray="3 4"/>
      <polygon points="${sx(0)},${sy(0)} ${pts} ${sx(1)},${sy(0)}" fill="#1B1B1F" opacity="0.08"/>
      <polyline points="${pts}" fill="none" stroke="#1B1B1F" stroke-width="2" stroke-linejoin="round"/>
      <text x="${sx(0.5)}" y="${S - 6}" text-anchor="middle">Good loans flagged</text>
      <text transform="translate(10 ${sy(0.5)}) rotate(-90)" text-anchor="middle">Bad loans caught</text>
      <text x="${sx(0.97)}" y="${sy(0.1)}" text-anchor="end" style="fill:#1B1B1F;font-weight:600">AUC ${m.test_auc.toFixed(3)}</text>
    </svg>`;
}

// ------------------------------------------------------------------ mobile dock
function watchDock() {
  const dock = $("#dock");
  const sheet = $("#decision");
  const footer = document.querySelector(".footer");
  let sheetVisible = true, footerVisible = false;
  const set = () => dock.classList.toggle("is-visible", !sheetVisible && !footerVisible);
  new IntersectionObserver(([e]) => { sheetVisible = e.isIntersecting; set(); }, { threshold: 0.05 }).observe(sheet);
  new IntersectionObserver(([e]) => { footerVisible = e.isIntersecting; set(); }).observe(footer);
}

// ------------------------------------------------------------------ start
async function start() {
  const res = await fetch("scorecard.json");
  if (!res.ok) throw new Error(`Could not load scorecard.json (${res.status})`);
  card = await res.json();
  state = readUrl();
  if (state.months_since_last_delinquency < NEVER) lastMonths = state.months_since_last_delinquency;
  buildBand();
  buildGrid();
  buildPerformance();
  syncForm();
  render();
  watchDock();
}

start().catch((err) => {
  $("#decision").innerHTML =
    `<p><strong>The scorecard didn't load.</strong> Run <code>python src/export_web.py</code> to create
     <code>site/scorecard.json</code>, then serve the folder with <code>python -m http.server -d site</code>.</p>`;
  console.error(err);
});
