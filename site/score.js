// Scores an applicant by looking up points in the table exported by
// src/export_web.py. No model logic lives here, so the browser can't drift
// from the Python scorecard. tests/test_web_parity.py checks they agree.

export function binIndex(feature, value) {
  if (feature.type === "categorical") {
    const i = feature.categories.indexOf(String(value));
    return i === -1 ? feature.categories.length - 1 : i;
  }
  const e = feature.edges; // null means +/- infinity
  for (let i = 0; i < e.length - 1; i++) {
    const lo = e[i] === null ? -Infinity : e[i];
    const hi = e[i + 1] === null ? Infinity : e[i + 1];
    if ((value > lo || i === 0) && value <= hi) return i;
  }
  return e.length - 2;
}

export function scoreApplicant(card, applicant) {
  const { offset, factor, score_min, score_max, approve_max_pd, refer_max_pd } = card.meta;
  const factors = card.features.map((f) => {
    const bin = binIndex(f, applicant[f.key]);
    const points = f.bins[bin].points;
    const best = f.bins[f.best].points;
    return { feature: f, bin, points, best, lost: best - points };
  });

  const raw = factors.reduce((sum, x) => sum + x.points, 0);
  const score = Math.round(Math.min(Math.max(raw, score_min), score_max));
  const pd = 1 / (1 + Math.exp((raw - offset) / factor));
  const g = card.grades.find((x) => pd < x.max_pd) || card.grades[card.grades.length - 1];
  const decision = pd < approve_max_pd ? "Approve" : pd < refer_max_pd ? "Refer" : "Decline";

  return { raw, score, pd, grade: g.grade, gradeLabel: g.label, decision, factors };
}
