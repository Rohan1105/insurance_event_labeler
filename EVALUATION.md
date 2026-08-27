# Evaluation

**Run configuration.** `python classifier.py && python evaluate.py` with a live
`GROQ_API_KEY` (loaded from `.env`), escalating to **`openai/gpt-oss-120b`** at
`temperature=0`. Of the 40 snippets, **28 were decided by rules and 12 by the LLM, with 0
falling back** — the offline fallback scorer did not fire at all. The `route` column in
`predictions.csv` shows which stage decided each item.

---

## 1. The numbers, and which one I care about

| metric | value |
|---|---|
| **Recall (YES)** | **0.944** — 17 of 18 real loss events caught |
| Precision (YES) | 0.895 — 17 of 19 flagged items were real |
| F1 (YES) | 0.919 |
| Accuracy | 0.925 (37/40) |
| Macro F1 | 0.925 |

| | pred YES | pred NO |
|---|---|---|
| **true YES** | 17 (TP) | **1 (FN — missed event, id 18)** |
| **true NO** | 2 (FP — analyst noise, ids 09 and 21) | 20 (TN) |

**Recall on YES is the headline**, because the two errors are not worth the same. A false
positive costs an analyst a few seconds to dismiss. A false negative is an event the insurer
never hears about — the exposure is live either way, and the only thing missing is the
warning. Accuracy hides this entirely: at a realistic base rate it would be dominated by the
NO class, and a classifier that answered NO to everything would still score well.

**These numbers are optimistic and should not be quoted as performance.** I wrote the rule
lexicons with all 40 snippets in front of me. This is a fit on the data, not a held-out
estimate. With 40 items, one flip moves accuracy by 2.5 points, and the Wilson 95% interval
on 37/40 runs roughly 80–98% — so "92%" and "82%" are not distinguishable at this sample
size. The honest claim is: *the approach is not obviously broken*, nothing more.

**One label was revised after the classifier ran.** Snippet 09 (single-eyewitness mine
collapse) moved from YES to NO on re-reading — see Case 3. Revising ground truth after seeing
predictions is a contamination risk worth naming explicitly, so: the revision moved *away*
from the classifier's answer and **cost 2.5 points of accuracy and 5 points of precision**
(38/40 → 37/40). Nobody games a metric downwards. It also forced a rubric rule that did not
exist before, which is the outcome I would want from the exercise.

### A 120-billion-parameter model changed the headline by exactly zero

I first ran this offline, where the 12 escalated items were resolved by a deliberately crude
keyword scorer (`weak_score_fallback`: count event and impact words, subtract topic and
future words, threshold at +1). Swapping that for `gpt-oss-120b`:

| | escalation = crude scorer | escalation = gpt-oss-120b |
|---|---|---|
| Accuracy | 0.925 (37/40) | 0.925 (37/40) |
| Recall (YES) | 0.944 | 0.944 |
| Precision (YES) | 0.895 | 0.895 |
| Escalated items correct | 10/12 | 10/12 |
| **Which items were wrong** | **09, 18, 21** | **09, 18, 21** |
| **Confidence on snippet 21** | **LOW** | **HIGH** |

Identical headline metrics, the same three items wrong — and the LLM was *more confident*
about one of them. This is the clearest evidence in the whole exercise that **the headline
metric is not measuring what I care about at this sample size.** If a real model and a
word-counter are indistinguishable on 40 items, the evaluation set is doing no work in
telling them apart, and any decision made on the strength of that comparison would be
unfounded. What actually differs between the two configurations is the *quality of the
confidence signal and the rationales* — neither of which appears in accuracy, precision,
recall, or F1.

The LLM is not thereby useless: the escalated items are exactly the ambiguous ones, and on
those it produced defensible one-line rationales that the word-counter cannot. But I would
need a larger, harder, held-out set to demonstrate the difference numerically, and I do not
have one.

### Where the decisions were made

| route | n | correct |
|---|---|---|
| `rule:event-with-impact` | 9 | 9 |
| `rule:future-framing` | 8 | 7 |
| `rule:non-event-topic` | 5 | 5 |
| `rule:non-physical-cause` | 4 | 4 |
| `rule:routine` | 2 | 2 |
| `llm` | 12 | 10 |

The split works as intended: the fast path took 28/40 (70%) and escalated 12. **The LLM
independently agreed with me about which items are hard.** It returned LOW confidence on
09, 15, 25, 26, 29, 33 and 39; my own LOW set is 09, 11, 15, 18, 20, 25, 26, 33, 39. Six of
its seven LOWs are items I also found hard. Since it never saw my labels, that overlap is
the closest thing to an independent difficulty signal in this exercise, and it is the main
reason I trust the escalation boundary.

Note that both remaining LLM errors are items it flagged as *hard* — 09 at LOW, 21 at
HIGH-but-unstable (below). Its uncertainty is landing in the right places even where its
answers are wrong, which is worth more operationally than the two points of accuracy.

### Right answer, wrong reason

Accuracy flatters the rules further. Auditing each of the 28 rule firings against the rubric
category I would have assigned by hand, **4 were correct by coincidence**:

| id | snippet subject | rule that fired | why it is the wrong reason |
|---|---|---|---|
| 04 | strike ballot next month | `future-framing` | It is a labour dispute. It matched only because "next month" is in the future lexicon. |
| 19 | cyber premium forecast | `non-physical-cause` | It is a market forecast. It matched on the word "cyber". |
| 38 | data breach disclosure | `routine` | It is a cyber item. It matched on "no misuse" before the cyber rule could fire. |
| 40 | sporting body requirements | `future-framing` | It is sport/policy. It matched on "bidding for future". |

So the defensible rule count is 23/28, not 27/28. On slightly different wording, all four
flip. This is invisible in any headline metric — I only found it by reading the `route`
column against the snippets.

### Confidence calibration

| classifier says | n | accuracy |
|---|---|---|
| HIGH | 33 | 0.939 |
| LOW | 7 | 0.857 |

The signal now points the right way — HIGH outperforms LOW — but only just, and with n=7 in
the LOW bucket the gap is not statistically meaningful. Two of the three errors are still
HIGH-confidence, which is the shape that matters operationally: HIGH is exactly what a
production system would auto-action without review.

The deeper problem is structural, not statistical: rule-path confidence is the hard-coded
constant `HIGH` in `fast_path`, so it expresses "a rule matched", not "this is likely
correct". Those are different claims and the code conflates them. Until that is fixed, the
HIGH/LOW output should be treated as unvalidated.

Against *my* annotation confidence the shape is as expected and much cleaner — the
classifier got 30/31 of my HIGH items (0.968) but only 7/9 of my LOW items (0.778). **My
uncertainty predicts the classifier's errors better than the classifier's own does**, which
is an argument for keeping a human-calibrated difficulty signal in the loop rather than
trusting the model's self-report.

### Run-to-run stability

`temperature=0` makes sampling greedy but does not guarantee determinism — batched GPU
inference can reorder floating-point reductions and break ties differently. I checked rather
than assumed (`python stability.py --runs 5`):

- **Labels: perfectly stable.** All 40 labels identical in every run — so accuracy, recall and
  precision were 0.925 / 0.944 / 0.895 in all five runs.
- **Confidence: not stable.** Exactly one item flickered — **id 21**, alternating between
  YES/HIGH and YES/LOW (LOW in 3 of 5 runs; reproduced over 3 further runs). This moves the
  calibration table between runs (accuracy on HIGH 0.939 ↔ 0.969) while leaving the headline
  untouched.

Two things follow. First, the headline metrics here are reproducible and I can quote them
without hedging about sampling. Second, **the one unstable item is the one my rubric is
self-contradictory about** — see Case 2. The model's uncertainty is landing precisely where
the specification is ambiguous, which is a point in its favour and a point against my rubric.

---

## 2. (a) Three disagreements

There are exactly three label disagreements — 09, 18 and 21 — so all three are analysed below.
The pattern across them is the finding, not the individual verdicts: **only one of the three
is a defect in the classifier. The other two are defects in my rubric**, which the classifier
exposed by applying it more literally than I ever did.

### Case 1 — id 18, cargo plane emergency landing. **Classifier wrong.**

> "A cargo plane made an emergency landing in Anchorage after an engine fire warning; the
> aircraft landed safely with no injuries."

Mine: **YES (LOW)**. Classifier: **NO (HIGH)**, via `rule:future-framing`.

**I am right, and I can show the exact mechanism.** The rule fired because "warning" is in
the future-framing lexicon and the impact evidence was empty — "no injuries" was correctly
stripped by the negation check, and "emergency landing" is classed as an event term rather
than an impact term, so nothing survived to block the NO. The rule read "engine fire
**warning**" as a forecast of a future event.

It is not a forecast. The warning is a past-tense trigger that already caused a diversion;
the aircraft is on the ground in the wrong place. This is a live aviation claim: an engine
inspection at minimum, an unscheduled landing, cargo delay, and a plausible hull/engine loss.
That the outcome was benign is knowable only afterwards, which is exactly why the label is
YES-with-LOW-confidence rather than NO.

The critical detail is that **this item never reached the LLM.** The fast path resolved it
with HIGH confidence, so the escalation stage that exists precisely for hedged cases like
this one never saw it. HIGH also means no human reviews it. This is the failure mode the
whole design is meant to prevent, and it slipped through because a lexicon matched a word
rather than its tense — which is a strong argument for the change I propose in §c.3: veto
rules should escalate rather than decide whenever any event term is present.

### Case 2 — id 21, Alpine snowfall. **Classifier wrong — but my rubric is also at fault.**

> "Heavy snowfall closed several Alpine passes overnight; no incidents were reported and
> roads reopened by morning."

Mine: **NO (HIGH)**. Classifier: **YES (HIGH)**, via `llm`, with the rationale:
*"Heavy snowfall caused Alpine pass closures, an active emergency response despite no
reported injuries."*

**I am right about the label.** The snippet states its own answer twice — "no incidents were
reported" and "reopened by morning". There is no damage, no injury, no ongoing response, and
the disruption is over. An insurer has nothing to reserve against.

**But the rationale shows this was a rubric defect, not a model failure — and I have fixed
the rubric.** The version of `RUBRIC.md` that produced this run contained two rules that both
apply here, and did not say which wins:

- *"Precautionary response, damage unconfirmed → YES"* — a road closure is a precautionary
  response to a physical hazard.
- *"Routine / explicitly no-incident → NO"* — "no incidents were reported".

The model applied my precautionary rule faithfully and reached YES. I applied my routine rule
and reached NO. **It followed the rubric as written; the rubric was ambiguous, and I had
resolved the ambiguity silently in my head without ever writing the tie-break down.** The
stability check above is corroborating evidence: this is the *only* item whose confidence
flickers between runs, so the model was detecting the ambiguity even when it landed on HIGH.

How I know my side is the right resolution: the precautionary rule exists to catch a hazard
that is *still unfolding and may yet produce a loss* — a pipeline shut on a pressure anomaly
(15), a dam evacuated over cracks (33). Snippet 21 is resolved and closed, with the outcome
already stated. `RUBRIC.md` now carries the precedence rule explicitly:

> An explicit statement that *no incident occurred* — not merely that no one was hurt — plus
> a stated return to normal, beats the precautionary rule.

The "not merely that no one was hurt" clause is doing real work, and it is what makes this a
rule rather than a patch. It is precisely what separates 21 from **18** in Case 1: the
aircraft's snippet also ends benignly ("landed safely with no injuries"), but there the
incident *did* occur — the diversion is the event — and only the harm was nil. A tie-break
phrased as "benign outcome → NO" would have flipped 18 to NO and broken the more important
of my two errors. Writing the rule forced me to find that distinction; I had not articulated
it before the classifier disagreed with me.

**What I did not change, on purpose:**

- **The LLM's system prompt is frozen** at the pre-fix rubric (see the comment in
  `llm_client.py`). Copying the new precedence rule into the prompt would very likely fix
  snippet 21 and produce 39/40 — but that number would be bought by editing the prompt after
  seeing which single item the model failed, on the only 40 examples I have. That is tuning on
  the test set. The rule propagates to the prompt on the next run against held-out data.
- **The rule path is unchanged.** R1 requires a routine marker *and* no impact terms, but
  "closed" is in the impact lexicon, so R1 was blocked and the item escalated at all. "Closed"
  is a bad impact term — closure is ambiguous between "damage forced it" and "precaution,
  reopened fine". Also a one-line fix, also not made, for the same reason.

So the reported metrics still stand at 37/40 with this item wrong, and I would rather submit
that than a tuned score. The distinction I am drawing: **fixing the specification is
legitimate and I did it; fixing the classifier against the specific item it failed is not.**
This is also the *cheap* error — a wasted analyst glance, not a missed event.

### Case 3 — id 09, Chilean mine collapse. **I am right, but the classifier was obeying my own rubric.**

> "Local media report a partial collapse at a copper mine in northern Chile; the operator has
> not confirmed casualties; a single eyewitness is cited."

Mine: **NO (LOW)**. Classifier: **YES (LOW)**, via `llm`, with the rationale: *"Single-source
report of a mine collapse constitutes a physical event with damage, though details are
unconfirmed."*

**This is the disagreement I changed my own mind on**, and the honest account matters. My
first pass labelled it YES/LOW, straight off the rubric's unconfirmed-report rule. On
re-reading I moved it to NO/LOW, and the classifier's rationale is what made the reason
legible: it says "single-source report... constitutes a physical event", and the word doing
the work is *constitutes*. It does not. A single-source report constitutes a **claim** that
an event occurred.

The distinguishing fact is **who characterised the event, and how reliably**. "Partial
collapse" is not an observation, it is a *judgement about severity* — and here it rests on one
untrained observer. Someone who sees a rockfall, some slippage, or a section of debris may
reasonably describe it as a partial collapse; the error runs upward, because a lone witness to
something alarming tends to report the alarming reading. The second leg is the operator's
silence: a mine operator would know whether its own workings had partially collapsed and would
be obliged to confirm casualties. Neither leg is decisive alone. Together, what is unverified
is not the casualty count but **the event description itself**.

Contrast id 39, which I kept at YES/LOW: flash floods reported by *multiple* local sources
with "details unverified". A flood is mass-observable and needs no expert characterisation —
nobody mistakes a minor puddle for a flash flood that sweeps a market town. There the *event
category* is not in doubt and only the casualty figures are soft. **My rubric had collapsed
these two situations into one rule** — "unconfirmed / single-source → YES, LOW" — which is why
the classifier answered YES and was right to, given what it was told. `RUBRIC.md` now
separates them:

> **Sourcing is judged on the event, not the numbers**: did the event *category* need a
> judgement call? Floods from multiple sources → YES; a "partial collapse" from one
> observer → NO.

The line is therefore not "one source vs. many". It is **whether the claim needs a judgement
call to make, and whether the person making it was in a position to make it.**

**How I know I am right rather than just inconsistent**: the rule generalises, and it
generalises in a direction that costs me. It reclassifies 09 as NO while leaving 29 (cause
unconfirmed, emergency services on scene → corroborated) and 39 untouched, and it drops my
measured accuracy from 38/40 to 37/40. A rule invented to flatter the numbers would not do
that.

**The honest caveats**, in the order I expect them to be raised:

1. **Operator silence is the weaker of my two legs, and may point the other way.** An operator
   has systematic legal and PR incentives to delay confirming casualties, so early
   non-confirmation is close to uninformative — arguably it is what you would expect *if the
   incident were serious*. I would not defend this leg on its own. The argument rests on the
   characterisation problem; the operator's silence only means no better-placed account has
   displaced the eyewitness yet.
2. **"Witness error skews high" is an assumption, not a measured fact.** I believe a lone
   observer over-reads an alarming scene more often than they under-read it, but I have no
   data here, and the opposite bias is arguable for an industrial site where access is
   restricted and the observer may see only part of it. This is precisely the kind of rubric
   premise that should be tested against claims outcomes (§b) rather than asserted.
3. **It cuts against my own §c argument** that false negatives are costlier. If the collapse is
   real, NO is exactly the expensive mistake. My defence is that a source-credibility floor is
   not a recall concession — a claim requiring an unreliable judgement call is not yet a missed
   *event* — but I concede the binary is doing damage here. The operational answer is a
   **"developing / unverified" tier** that re-scores as corroboration arrives, which does not
   exist in this build. That is a limitation of the design, not a virtue of the label.

If pushed hard on any of these, the position I would actually defend is narrower than NO: **09
belongs in a watchlist, and the rubric forces me to call it something.**

### A fourth disagreement, on confidence only — id 20

Worth recording because it is the one with the largest production consequences, even though
the labels agree. On the ransomware snippet, mine is **NO (LOW)**, the classifier's is **NO
(HIGH)** via `rule:non-physical-cause`.

My NO is conditional on a scope assumption — that this feed serves P&C rather than cyber —
which is an input from the client, not a fact about the snippet. If the insurer writes cyber
or contingent business interruption, this is a large loss across three plants and the label
flips to YES. My LOW carries that conditionality forward; the classifier's HIGH destroys it,
asserting as certain something true only under an unstated premise. And it is not reasoning:
the same rule produced the right answer on id 19 (a *market forecast*) purely because the
word "cyber" appears, and `HIGH` is a literal constant in the code. The fix is structural —
cyber/BI should be a third output class routed by client policy, not hard-coded as a NO
inside a rule.

---

## 2. (b) What is wrong with using my own labels as ground truth

**They are not ground truth. They are one annotator's opinion, and the classifier was built
by the same person on the same day.**

**1. One annotator means there is nothing to compare against.** With a single labeller there
is no agreement to measure, so I cannot separate real signal from my own habits. My 9
LOW-confidence labels are where a second person would most likely disagree — that is 22% of
the set.

**2. I wrote the rubric, the labels and the classifier.** So agreement partly measures me
agreeing with myself. The evidence is direct: **two of my three errors turned out to be
defects in my rubric, not the model** (Cases 2 and 3). Both only surfaced when something
outside my head applied the rubric literally.

**3. I changed a label after seeing the predictions.** Snippet 09. I argue in Case 3 that it
was a genuine correction — it *cost* me accuracy, and the rule it produced generalises — but
"the annotator revised a label after seeing the model's answer" is exactly the pattern a
reviewer should distrust, and my own assurance does not fix it. The fix is procedural: freeze
and version labels before generating predictions.

**4. Nothing here is checked against real claims.** My rubric's judgement calls — precautionary
responses → YES, single-source → NO — are assumptions about business value, not measured
facts. Snippet 15's pipeline shutdown may routinely produce zero claims; single-source mine
collapses may routinely turn out true. If so, the rubric is wrong and every metric built on it
inherits the error.

**5. 40 items is too few, and too evenly split.** 18 YES / 22 NO is nothing like a real feed,
where YES might be 1%. Precision measured at 45% prevalence says almost nothing about
precision at 1%. The proof is in §1: this set cannot even tell a 120B model apart from a
word-counter, so it cannot be measuring much.

**What I would do in production:**

- **Two or three annotators on the same sample, and measure agreement** (Cohen's κ). Where
  agreement is low, rewrite the *rubric* — low agreement means the spec is ambiguous, not that
  the labellers are bad. Case 2 proves it: that item would have split annotators, and the right
  response was the missing tie-break rule.
- **Blind labelling and a frozen held-out set.** Annotators never see model output, whoever
  wrote the classifier does not label the evaluation set, and labels are versioned before
  predictions exist. That closes problems 1, 2 and 3 together.
- **Check against actual claims after 6–12 months.** Did the items we flagged YES produce
  claims? Did claims arrive from items we called NO? That is the only genuinely external
  answer, and the only way to find out whether the rubric's judgement calls were right.

---

## 2. (c) At 100,000 items a day: which error matters more, and what to tune

**False negatives matter more, by a wide margin, and I would tune for recall.**

The asymmetry is in the cost, not the count. A false positive costs an analyst a few seconds
to dismiss — a bounded, linear, cheap cost that scales with staffing. A false negative costs
the insurer knowledge of a live event: no early reserve, no reinsurance notification, no
loss adjuster dispatched, first notice arriving from the claimant instead of the feed. The
downside is unbounded and correlated — the events most likely to be missed early (an
unconfirmed report from a hard-access region, id 39; a dam showing cracks, id 33) are exactly
the ones where early warning is worth most.

The volume argument runs the same way. At 100,000/day with a 1% base rate, ~1,000 real events:

- At today's 94.4% recall, **~56 real events are missed every day**.
- Pushing recall to 0.98 and letting precision fall to 0.80 takes false positives from
  ~111/day to ~245/day: **~134 extra items to dismiss, to catch ~36 more real events.**
- At a few seconds each, 134 dismissals is well under an analyst-hour. 36 missed events is
  not.

Precision at a 1% base rate will be *far* worse than the 0.895 measured here whatever I do —
that is arithmetic, not a tuning failure, and it is why the operational answer is a ranked
queue rather than a binary flag.

**What I would tune, in order:**

**1. Output a score, not a YES/NO.** This is the real limitation of the current build: the
rules are hard-coded and the LLM returns a bare enum, so **there is no threshold to move at
all**. With a continuous score I can pick the operating point deliberately on a
precision-recall curve instead of accepting whatever the code happens to do.

**2. Set that threshold by a recall target, not by F1.** Pick the lowest threshold the analyst
queue can absorb — "recall ≥ 0.98, whatever precision that implies" — then staff to the
resulting volume. F1 treats both errors as equally bad, which is exactly the assumption I have
just argued against. If one number is needed, F2 weights recall 4× more than precision.

**3. Make the NO rules escalate instead of decide.** Every fast-path NO rule is an
unreviewable false-negative generator, and snippet 18 was lost to one — it never reached the
LLM that exists to catch precisely that case. I would escalate whenever any event word is
present. This costs money at 100,000/day, so it is a trade to measure rather than assume.

**4. Add tiers instead of a binary.** Auto-drop, analyst queue, and alert — plus a
**"developing / unverified"** lane for things like snippet 09, which is neither a confirmed
event nor nothing, and a separate **"watch" feed** for forecasts and warnings (07, 14, 23).
Those are correctly NO today, but they are commercially valuable and deserve their own
channel rather than being noise. This makes the classifier's job *ranking* rather than
*deciding*, which is a much easier job to do well.

**5. Sample the items we drop.** Audit a random 1% of auto-dropped items each day against
human labels. By definition nobody ever sees a false negative in production, so this is the
only way to measure the error type I have just claimed matters most — about 1,000 items/day,
enough to catch a recall regression within a week.

One thing I would *not* do yet: build any of this on the current confidence signal. It
separates HIGH from LOW by 8 points on n=7, and on the rule path it is a hard-coded constant
rather than an estimate. Two of the three errors are HIGH-confidence, so tiering on it today
would auto-action exactly the wrong items. It needs calibrating against held-out data first.

**Operational note from this run:** the free tier rate-limited at 8,000 tokens/minute, and the
system prompt is ~520 tokens of the ~700 per call. At 100,000 items/day with a 30% escalation
rate that is ~30,000 LLM calls/day, and the prompt would dominate the bill. Batching several
snippets per call, or caching the system prompt, is the first cost optimisation — and it is
another reason the fast path earns its place.
