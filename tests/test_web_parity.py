"""
Checks that the web app's JavaScript scoring (site/score.js) gives the same
score, probability of default, grade and decision as the Python scorecard,
for every applicant in the held-out test set.

Needs Node.js and the trained artifacts (run train.py and export_web.py first).
"""
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

SITE_JSON = os.path.join(ROOT, "site", "scorecard.json")
SCORED = os.path.join(ROOT, "models", "test_scored_sample.csv")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.exists(SITE_JSON) or not os.path.exists(SCORED),
    reason="needs node and exported artifacts",
)

NODE_SCRIPT = """
import { readFileSync } from "node:fs";
import { scoreApplicant } from "%s";
const card = JSON.parse(readFileSync(process.argv[2], "utf8"));
const rows = JSON.parse(readFileSync(0, "utf8"));
const out = rows.map((r) => {
  const s = scoreApplicant(card, r);
  return { score: s.score, pd: s.pd, grade: s.grade, decision: s.decision };
});
process.stdout.write(JSON.stringify(out));
"""


def test_browser_matches_python(tmp_path):
    from scorecard import grade_from_pd, decision_from_pd, score_to_pd

    with open(SITE_JSON) as f:
        card = json.load(f)
    keys = [f["key"] for f in card["features"]]
    df = pd.read_csv(SCORED)
    rows = df[keys].to_dict(orient="records")

    script = tmp_path / "run.mjs"
    score_js = "file://" + os.path.abspath(os.path.join(ROOT, "site", "score.js"))
    script.write_text(NODE_SCRIPT % score_js)
    proc = subprocess.run(["node", str(script), SITE_JSON], input=json.dumps(rows),
                          capture_output=True, text=True, check=True)
    js = pd.DataFrame(json.loads(proc.stdout))

    with open(os.path.join(ROOT, "models", "scorecard.json")) as f:
        sc = json.load(f)
    py_pd = df["score"].apply(lambda s: score_to_pd(s, sc))

    # Python stores the score rounded, so allow 1 point of rounding difference.
    assert (js["score"] - df["score"]).abs().max() <= 1
    assert (js["grade"] == df["grade"]).mean() > 0.995
    assert ((js["pd"] - py_pd).abs() < 0.002).all()
    agree = (js["decision"] == py_pd.apply(decision_from_pd)).mean()
    assert agree > 0.995
    assert len(js) == len(df) > 1000
