"""Evaluate predictions.csv against labels.csv.

Metrics are implemented directly rather than pulled from scikit-learn: there are only
40 rows, the definitions are four lines each, and hand-rolling them keeps the repo
dependency-free and makes the positive-class choice explicit.

YES is the positive class throughout, because the expensive error is a missed event.

Run:  python evaluate.py --truth labels.csv --pred predictions.csv --report reports/eval_report.md
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

POSITIVE = "YES"
NEGATIVE = "NO"


@dataclass
class Row:
    id: str
    truth: str
    truth_conf: str
    truth_just: str
    pred: str
    pred_conf: str
    pred_just: str
    route: str
    text: str


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def counts_for(rows: list[Row], positive: str) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) treating `positive` as the positive class."""
    tp = sum(1 for r in rows if r.truth == positive and r.pred == positive)
    fp = sum(1 for r in rows if r.truth != positive and r.pred == positive)
    fn = sum(1 for r in rows if r.truth == positive and r.pred != positive)
    tn = sum(1 for r in rows if r.truth != positive and r.pred != positive)
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1. Zero denominators give 0.0, the usual convention."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def accuracy(rows: list[Row]) -> float:
    return sum(1 for r in rows if r.truth == r.pred) / len(rows) if rows else 0.0


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_csv(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["id"].strip(): row for row in csv.DictReader(fh)}


def load_rows(truth_path: str, pred_path: str, snippets_path: str) -> list[Row]:
    truth = _read_csv(truth_path)
    pred = _read_csv(pred_path)
    texts = {k: v["text"] for k, v in _read_csv(snippets_path).items()}

    missing = set(truth) ^ set(pred)
    if missing:
        raise SystemExit(f"id mismatch between truth and predictions: {sorted(missing)}")

    rows = []
    for sid in sorted(truth):
        t, p = truth[sid], pred[sid]
        rows.append(
            Row(
                id=sid,
                truth=t["label"].strip().upper(),
                truth_conf=t["confidence"].strip().upper(),
                truth_just=(t.get("justification") or "").strip(),
                pred=p["label"].strip().upper(),
                pred_conf=p["confidence"].strip().upper(),
                pred_just=(p.get("justification") or "").strip(),
                route=(p.get("route") or "").strip(),
                text=texts.get(sid, ""),
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def build_report(rows: list[Row]) -> str:
    tp, fp, fn, tn = counts_for(rows, POSITIVE)
    p_yes, r_yes, f_yes = prf(tp, fp, fn)
    # For the NO class the roles of fp/fn swap.
    p_no, r_no, f_no = prf(tn, fn, fp)
    macro_f1 = (f_yes + f_no) / 2
    macro_p = (p_yes + p_no) / 2
    macro_r = (r_yes + r_no) / 2

    out: list[str] = []
    add = out.append

    add("## Headline numbers\n")
    add(f"- **Recall (YES) — the number that matters: {r_yes:.3f}** "
        f"({tp}/{tp + fn} real loss events caught)")
    add(f"- Precision (YES): {p_yes:.3f} ({tp}/{tp + fp} flagged items were real events)")
    add(f"- F1 (YES): {f_yes:.3f}")
    add(f"- Accuracy: {accuracy(rows):.3f} ({sum(1 for r in rows if r.truth == r.pred)}/{len(rows)})")
    add(f"- Macro F1: {macro_f1:.3f} (macro precision {macro_p:.3f}, macro recall {macro_r:.3f})")
    add("")

    add("## Per-class\n")
    add("| class | precision | recall | F1 | support |")
    add("|---|---|---|---|---|")
    add(f"| YES | {p_yes:.3f} | {r_yes:.3f} | {f_yes:.3f} | {tp + fn} |")
    add(f"| NO | {p_no:.3f} | {r_no:.3f} | {f_no:.3f} | {tn + fp} |")
    add("")

    add("## Confusion matrix\n")
    add("| | pred YES | pred NO |")
    add("|---|---|---|")
    add(f"| **true YES** | {tp} (TP) | {fn} (FN — missed events) |")
    add(f"| **true NO** | {fp} (FP — analyst noise) | {tn} (TN) |")
    add("")

    add("## Where the decision was made\n")
    add("| route | n | correct | accuracy |")
    add("|---|---|---|---|")
    for route in sorted({r.route for r in rows}):
        subset = [r for r in rows if r.route == route]
        correct = sum(1 for r in subset if r.truth == r.pred)
        add(f"| `{route}` | {len(subset)} | {correct} | {correct / len(subset):.3f} |")
    add("")

    add("## Confidence calibration\n")
    add("Does the classifier's own HIGH/LOW signal predict when it is right?\n")
    add("| classifier confidence | n | correct | accuracy |")
    add("|---|---|---|---|")
    for conf in ("HIGH", "LOW"):
        subset = [r for r in rows if r.pred_conf == conf]
        if not subset:
            continue
        correct = sum(1 for r in subset if r.truth == r.pred)
        add(f"| {conf} | {len(subset)} | {correct} | {correct / len(subset):.3f} |")
    add("")
    add("Same question against *my* annotation confidence — how hard were the items I "
        "found hard?\n")
    add("| my confidence | n | classifier correct | accuracy |")
    add("|---|---|---|---|")
    for conf in ("HIGH", "LOW"):
        subset = [r for r in rows if r.truth_conf == conf]
        if not subset:
            continue
        correct = sum(1 for r in subset if r.truth == r.pred)
        add(f"| {conf} | {len(subset)} | {correct} | {correct / len(subset):.3f} |")
    add("")

    disagreements = [r for r in rows if r.truth != r.pred]
    add(f"## Label disagreements ({len(disagreements)})\n")
    if not disagreements:
        add("_None._\n")
    else:
        add("| id | mine | classifier | route | snippet |")
        add("|---|---|---|---|---|")
        for r in disagreements:
            kind = "FN" if r.truth == POSITIVE else "FP"
            add(f"| {r.id} | {r.truth} ({r.truth_conf}) | {r.pred} ({r.pred_conf}) — {kind} "
                f"| `{r.route}` | {r.text} |")
        add("")

    conf_only = [r for r in rows if r.truth == r.pred and r.truth_conf != r.pred_conf]
    add(f"## Confidence-only disagreements ({len(conf_only)})\n")
    add("Same label, different certainty. Not errors, but they show where the two "
        "notions of confidence come apart.\n")
    if conf_only:
        add("| id | label | mine | classifier | route |")
        add("|---|---|---|---|---|")
        for r in conf_only:
            add(f"| {r.id} | {r.truth} | {r.truth_conf} | {r.pred_conf} | `{r.route}` |")
        add("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier predictions against labels.")
    parser.add_argument("--truth", default="labels.csv")
    parser.add_argument("--pred", default="predictions.csv")
    parser.add_argument("--snippets", default="snippets.csv")
    parser.add_argument("--report", default="reports/eval_report.md")
    args = parser.parse_args(argv)

    rows = load_rows(args.truth, args.pred, args.snippets)
    report = build_report(rows)
    print(report)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# Evaluation report (generated by evaluate.py)\n\n" + report + "\n")
        print(f"\n[written] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
