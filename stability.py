"""Run the classifier N times and report what changes between runs.

`temperature=0` is not a determinism guarantee -- it makes sampling greedy, but batched
GPU inference still reorders floating-point reductions, so ties can break differently on
identical input. If a metric moves between runs, that variance belongs in the report
alongside the metric.

Costs N x (number of escalated snippets) LLM calls, so it is a separate script rather
than part of the default pipeline.

Run:  python stability.py --runs 5
"""

from __future__ import annotations

import argparse
from collections import Counter

import llm_client
from classifier import classify_all, load_snippets
from evaluate import Row, accuracy, counts_for, prf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure run-to-run stability of the classifier.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--snippets", default="snippets.csv")
    parser.add_argument("--truth", default="labels.csv")
    args = parser.parse_args(argv)

    snippets = load_snippets(args.snippets)
    truth = {}
    import csv

    with open(args.truth, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            truth[row["id"].strip()] = row["label"].strip().upper()

    llm = llm_client.from_env()
    if llm is None:
        print("[warn] no LLM configured -- the rule path is deterministic, so this "
              "will trivially report 100% stability.")

    runs: list[dict[str, tuple[str, str]]] = []
    for i in range(args.runs):
        labels = classify_all(snippets, llm)
        runs.append({lab.id: (lab.label, lab.confidence) for lab in labels})
        rows = [
            Row(lab.id, truth[lab.id], "", "", lab.label, lab.confidence, "", lab.route, "")
            for lab in labels
        ]
        tp, fp, fn, _ = counts_for(rows, "YES")
        precision, recall, _ = prf(tp, fp, fn)
        print(f"run {i + 1}: accuracy={accuracy(rows):.3f} recall={recall:.3f} "
              f"precision={precision:.3f}")

    print("\nItems that were not identical in every run:")
    label_flips = conf_flips = 0
    for sid in sorted(runs[0]):
        outcomes = [r[sid] for r in runs]
        if len(set(outcomes)) == 1:
            continue
        if len({o[0] for o in outcomes}) > 1:
            label_flips += 1
        else:
            conf_flips += 1
        tally = Counter(f"{lab}/{conf}" for lab, conf in outcomes)
        print(f"  {sid}: " + ", ".join(f"{k} x{v}" for k, v in tally.most_common()))
    if not (label_flips or conf_flips):
        print("  none")

    print(f"\n{label_flips}/{len(snippets)} items flipped LABEL across {args.runs} runs "
          f"(this is what moves the headline metrics)")
    print(f"{conf_flips}/{len(snippets)} items flipped CONFIDENCE only "
          f"(this moves the calibration table but not accuracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
