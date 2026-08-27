"""Insurable-loss-event classifier.

Two-stage design, mirroring how a human triage desk actually works:

  Stage 1 (fast path)  cheap lexical rules that fire only on unambiguous cases and
                       return HIGH confidence. Deterministic, free, ~1ms.
  Stage 2 (escalation) anything stage 1 refuses to call is sent to an LLM with the
                       rubric in the system prompt, which returns label + confidence
                       + a one-line rationale.

The point of the split is that the classifier's own confidence signal means something:
HIGH means "a rule matched with no contradicting evidence", LOW means "this needed
judgement". That is deliberately a different notion of confidence from the one in
labels.csv (which is my certainty as an annotator), and the evaluation reports both.

Run:  python classifier.py --in snippets.csv --out predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import llm_client

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Snippet:
    id: str
    text: str


@dataclass(frozen=True)
class Label:
    id: str
    label: str  # "YES" | "NO"
    confidence: str  # "HIGH" | "LOW"
    justification: str  # required when confidence == "LOW"
    route: str  # which stage decided: "rule:<name>" | "llm" | "fallback"

    def __post_init__(self) -> None:
        if self.label not in ("YES", "NO"):
            raise ValueError(f"{self.id}: bad label {self.label!r}")
        if self.confidence not in ("HIGH", "LOW"):
            raise ValueError(f"{self.id}: bad confidence {self.confidence!r}")
        if self.confidence == "LOW" and not self.justification.strip():
            raise ValueError(f"{self.id}: LOW confidence requires a justification")


# --------------------------------------------------------------------------------------
# Lexicons
#
# Written from the rubric's categories, not from reading the 40 snippets one by one --
# but I did have the 40 in front of me, so treat stage-1 accuracy as a fit, not an
# out-of-sample estimate. See EVALUATION.md.
# --------------------------------------------------------------------------------------

# A physical thing that happened.
EVENT_TERMS = [
    "fire", "blaze", "wildfire", "wildfires", "explosion", "blast", "exploded",
    "flood", "floods", "flooding", "inundated", "storm", "storms", "hurricane",
    "typhoon", "cyclone", "tornado", "hail", "hailstorm", "hailstorms", "earthquake",
    "quake", "tsunami", "landslide", "mudslide", "avalanche", "eruption", "drought",
    "collapse", "collapsed", "derailed", "derailment", "aground", "grounding",
    "capsized", "sank", "sinking", "spill", "leak", "rupture", "ruptured", "crash",
    "crashed", "emergency landing", "failure", "failed", "burst", "subsidence",
]

# Evidence of loss, harm, or an emergency response in progress.
IMPACT_TERMS = [
    "destroyed", "damage", "damaged", "damages", "injured", "injuries", "hospitalised",
    "hospitalized", "killed", "dead", "deaths", "fatalities", "missing", "casualties",
    "evacuated", "evacuation", "rescue", "rescued", "trapped", "salvage", "towed",
    "emergency services", "firefighters", "shut down", "shutdown", "suspended",
    "offline", "without power", "submerged", "closed", "stranded", "contained",
]

# Subject matter that is money, words, or the future rather than a physical event.
NON_EVENT_TERMS = [
    "shares", "share price", "analysts", "downgraded", "upgraded its", "investors",
    "funding", "series c", "raised $", "ipo", "valuation", "premiums", "premium",
    "renewals", "brokers", "pricing", "underwriting", "underwrite", "reinsurer",
    "reinsurance", "insurtech", "court ordered", "court", "lawsuit", "settlement",
    "regulations", "regulation", "regulatory", "consultation", "legislation", "policy",
    "study", "research", "academic", "paper", "method", "strike ballot", "union",
    "pay deal", "administration", "filed for", "sustainability", "requirements",
    "announced", "sport", "sporting", "tournament", "championship",
]

# Framing that puts the event in the future rather than the present.
FUTURE_TERMS = [
    "forecast", "forecasts", "is forecast to", "expected to strengthen", "predicting",
    "predicts", "outlook", "warning", "warnings", "advisory issued", "likely to",
    "coming decades", "next year", "next month", "will rise", "would stop",
    "seasonal", "bidding for future",
]

# Explicit statements that nothing happened.
ROUTINE_TERMS = [
    "routine", "no incidents", "no incident", "reopened", "no misuse",
    "as normal", "precautionary maintenance",
]

# Non-physical causes -- cyber / IT. Real losses, wrong line of business for this feed.
NON_PHYSICAL_CAUSE_TERMS = [
    "ransomware", "cyber", "cyberattack", "data breach", "misconfigured", "database",
    "software", "it systems", "hacked", "malware", "phishing",
]

NEGATORS = {"no", "not", "without", "never", "denied", "nor"}

# Hedges that mean the event is reported but not established.
UNCERTAINTY_TERMS = [
    "unconfirmed", "not confirmed", "has not confirmed", "unverified", "not verified",
    "single eyewitness", "preliminary", "local media report", "local sources report",
    "reports suggest", "as a precaution", "precaution", "assessing", "under way",
    "so far", "details are unverified", "may be",
]


# --------------------------------------------------------------------------------------
# Lexical matching
# --------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps $ and % so '$80m' and '4%' survive as signals."""
    return re.findall(r"[a-z0-9$%]+", text.lower())


def _phrase_positions(tokens: list[str], phrase: str) -> list[int]:
    """Start indices where `phrase` (1+ words) occurs in `tokens`."""
    words = phrase.split()
    n = len(words)
    return [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == words]


def _is_negated(tokens: list[str], start: int, window: int = 3) -> bool:
    """True if a negator sits in the `window` tokens immediately before `start`.

    Catches 'no oil spill', 'no injuries were reported', 'has not confirmed casualties'
    without needing a parser. Multiword phrases that begin with their own negator
    (e.g. 'without power') are unaffected because we look strictly before the match.
    """
    return any(tok in NEGATORS for tok in tokens[max(0, start - window):start])


def find_terms(text: str, lexicon: Iterable[str], respect_negation: bool = True) -> list[str]:
    """Terms from `lexicon` present in `text`, dropping negated occurrences."""
    tokens = tokenize(text)
    hits = []
    for term in lexicon:
        positions = _phrase_positions(tokens, term)
        if not positions:
            continue
        if respect_negation and all(_is_negated(tokens, p) for p in positions):
            continue  # every occurrence was negated -> not evidence
        hits.append(term)
    return hits


# --------------------------------------------------------------------------------------
# Stage 1: rule fast path
# --------------------------------------------------------------------------------------


def fast_path(snippet: Snippet) -> Optional[Label]:
    """Return a HIGH-confidence Label, or None to escalate.

    Rules are ordered by how strongly they veto: an explicit "nothing happened"
    statement beats everything, then non-physical cause, then future framing, then
    topic, and only then the positive event rule. Any uncertainty hedge blocks the
    positive rule outright -- hedged cases are exactly what the LLM is for.
    """
    text = snippet.text
    events = find_terms(text, EVENT_TERMS)
    impacts = find_terms(text, IMPACT_TERMS)
    non_events = find_terms(text, NON_EVENT_TERMS)
    futures = find_terms(text, FUTURE_TERMS)
    routines = find_terms(text, ROUTINE_TERMS)
    cyber = find_terms(text, NON_PHYSICAL_CAUSE_TERMS)
    hedges = find_terms(text, UNCERTAINTY_TERMS, respect_negation=False)

    def no(rule: str) -> Label:
        return Label(snippet.id, "NO", "HIGH", "", f"rule:{rule}")

    # R1: explicitly characterised as routine / no incident.
    if routines and not impacts:
        return no("routine")

    # R2: cause is cyber/IT and no physical event term survives -> different line of business.
    if cyber and not events:
        return no("non-physical-cause")

    # R3: framed as a forecast/warning/outlook with no realised impact.
    if futures and not impacts:
        return no("future-framing")

    # R4: financial / legal / regulatory / research subject matter, no physical event.
    if non_events and not events:
        return no("non-event-topic")

    # R5: a physical event with concrete impact and nothing hedging it.
    if events and impacts and not hedges:
        return Label(snippet.id, "YES", "HIGH", "", "rule:event-with-impact")

    return None  # escalate


# --------------------------------------------------------------------------------------
# Stage 2 fallback (used only when no LLM is configured)
# --------------------------------------------------------------------------------------


def weak_score_fallback(snippet: Snippet) -> Label:
    """Last-resort scorer so the pipeline still runs with no API key.

    Deliberately crude and always LOW confidence: it exists so the evaluation is
    reproducible offline, not because it is a good classifier.
    """
    text = snippet.text
    score = (
        len(find_terms(text, EVENT_TERMS))
        + len(find_terms(text, IMPACT_TERMS))
        - len(find_terms(text, FUTURE_TERMS))
        - len(find_terms(text, NON_EVENT_TERMS))
    )
    label = "YES" if score >= 1 else "NO"
    return Label(
        snippet.id,
        label,
        "LOW",
        f"No LLM configured; keyword score {score:+d} -> {label}.",
        "fallback",
    )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def classify(snippet: Snippet, llm: Optional[llm_client.GroqClient] = None) -> Label:
    """Classify one snippet. Stage 1 if it is confident, otherwise stage 2."""
    decided = fast_path(snippet)
    if decided is not None:
        return decided

    if llm is None:
        return weak_score_fallback(snippet)

    verdict = llm.classify(snippet.text)
    if verdict is None:  # network/parse failure -- degrade, do not crash the batch
        reason = llm.last_error or "unknown error"
        print(f"[warn] {snippet.id}: LLM call failed ({reason}) -- using fallback", file=sys.stderr)
        fb = weak_score_fallback(snippet)
        return Label(fb.id, fb.label, "LOW", "LLM call failed; " + fb.justification, "fallback")

    label, confidence, rationale = verdict
    if confidence == "LOW" and not rationale:
        rationale = "LLM returned LOW confidence without a rationale."
    return Label(snippet.id, label, confidence, rationale, "llm")


def classify_all(snippets: Iterable[Snippet], llm=None) -> list[Label]:
    return [classify(s, llm) for s in snippets]


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


def load_snippets(path: str) -> list[Snippet]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Snippet(row["id"].strip(), row["text"].strip()) for row in csv.DictReader(fh)]


def write_predictions(path: str, labels: list[Label]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "label", "confidence", "justification", "route"])
        writer.writeheader()
        for lab in labels:
            writer.writerow(asdict(lab))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify news snippets as insurable loss events.")
    parser.add_argument("--in", dest="infile", default="snippets.csv")
    parser.add_argument("--out", dest="outfile", default="predictions.csv")
    parser.add_argument("--no-llm", action="store_true", help="skip escalation even if a key is set")
    args = parser.parse_args(argv)

    snippets = load_snippets(args.infile)
    llm = None if args.no_llm else llm_client.from_env()
    if llm is None:
        print("[warn] no LLM configured -- escalated cases use the weak fallback scorer.", file=sys.stderr)
    else:
        print(f"[info] escalation via Groq model {llm.model}", file=sys.stderr)

    labels = classify_all(snippets, llm)
    write_predictions(args.outfile, labels)

    routed = sum(1 for lab in labels if lab.route != "fallback" and not lab.route.startswith("rule:"))
    fallback = sum(1 for lab in labels if lab.route == "fallback")
    print(
        f"{len(labels)} snippets: {len(labels) - routed - fallback} by rule, "
        f"{routed} by LLM, {fallback} by fallback -> {args.outfile}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
