"""Smoke tests for the parts most likely to break silently.

Plain asserts, no pytest dependency:  python test_classifier.py
"""

from classifier import (
    Label,
    Snippet,
    fast_path,
    find_terms,
    tokenize,
    _is_negated,
    _phrase_positions,
)
from evaluate import Row, counts_for, prf, accuracy
from llm_client import _parse_verdict


def test_tokenize_keeps_symbols():
    assert tokenize("Shares fell 4% after a $80M round.") == [
        "shares", "fell", "4%", "after", "a", "$80m", "round"
    ]


def test_phrase_positions_matches_multiword():
    tokens = tokenize("the vessel has run aground off Sri Lanka")
    assert _phrase_positions(tokens, "aground") == [4]
    assert _phrase_positions(tokens, "run aground") == [3]
    assert _phrase_positions(tokens, "run ashore") == []


def test_negation_window():
    tokens = tokenize("no oil spill has been reported")
    spill = _phrase_positions(tokens, "spill")[0]
    assert _is_negated(tokens, spill)
    tokens2 = tokenize("a large oil spill was reported")
    assert not _is_negated(tokens2, _phrase_positions(tokens2, "spill")[0])


def test_find_terms_drops_negated_hits():
    # 'injuries' is present but negated, so it must not count as impact evidence.
    assert find_terms("no injuries were reported", ["injuries"]) == []
    assert find_terms("four injuries were reported", ["injuries"]) == ["injuries"]


def test_fast_path_unambiguous_yes():
    s = Snippet("t1", "A fire destroyed a warehouse; two workers were injured.")
    out = fast_path(s)
    assert out is not None and out.label == "YES" and out.confidence == "HIGH"


def test_fast_path_unambiguous_no():
    s = Snippet("t2", "Shares in a reinsurer fell after analysts downgraded the sector.")
    out = fast_path(s)
    assert out is not None and out.label == "NO"


def test_fast_path_escalates_on_hedge():
    # Real event type, but hedged -> must not be resolved by rules.
    s = Snippet("t3", "Local media report a partial collapse; casualties are unconfirmed.")
    assert fast_path(s) is None


def test_label_requires_justification_when_low():
    try:
        Label("t4", "YES", "LOW", "", "llm")
    except ValueError:
        return
    raise AssertionError("LOW confidence without justification should raise")


def test_parse_verdict_handles_fenced_json():
    raw = '```json\n{"label":"yes","confidence":"low","rationale":"unconfirmed"}\n```'
    assert _parse_verdict(raw) == ("YES", "LOW", "unconfirmed")
    assert _parse_verdict("I think it is a yes.") is None
    assert _parse_verdict('{"label":"MAYBE","confidence":"HIGH"}') is None


def _row(truth, pred):
    return Row("x", truth, "HIGH", "", pred, "HIGH", "", "rule:test", "")


def test_metrics_against_hand_worked_example():
    rows = [
        _row("YES", "YES"), _row("YES", "YES"), _row("YES", "NO"),   # 2 TP, 1 FN
        _row("NO", "YES"),                                            # 1 FP
        _row("NO", "NO"), _row("NO", "NO"),                           # 2 TN
    ]
    tp, fp, fn, tn = counts_for(rows, "YES")
    assert (tp, fp, fn, tn) == (2, 1, 1, 2)
    precision, recall, f1 = prf(tp, fp, fn)
    assert abs(precision - 2 / 3) < 1e-9
    assert abs(recall - 2 / 3) < 1e-9
    assert abs(f1 - 2 / 3) < 1e-9
    assert abs(accuracy(rows) - 4 / 6) < 1e-9


def test_prf_handles_empty_denominators():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
