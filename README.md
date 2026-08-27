# Insurable loss event classifier — Leadenhall Analytics take-home

Classifies short news snippets as **YES** (describes an event that could produce insurance
losses) or **NO**, with a HIGH/LOW confidence signal and a justification for every LOW.

| file | what it is |
|---|---|
| `RUBRIC.md` | The labelling rules, including explicit handling of the awkward cases. One page. |
| `labels.csv` | My labels for all 40 snippets — `id, label, confidence, justification`. Ground truth for the evaluation. |
| `snippets.csv` | The 40 snippets, transcribed from the assessment PDF. |
| `classifier.py` | The classifier: rule fast path + LLM escalation. |
| `llm_client.py` | Groq chat-completions client (stdlib only) + `.env` loader. |
| `evaluate.py` | Metrics, confusion matrix, disagreement analysis, markdown report. |
| `stability.py` | Runs the classifier N times and reports what changes between runs. |
| `test_classifier.py` | Smoke tests for negation handling, rule dispatch, JSON parsing, metrics. |
| `requirements.txt` | Empty by design — standard library only. The file records why. |
| `EVALUATION.md` | The numbers and the three written answers. |
| `predictions.csv`, `reports/eval_report.md` | Generated output, committed so results are inspectable without a run. |

## How to run

Python 3.10+. **No dependencies** — standard library only.

```bash
python test_classifier.py                  # 11 smoke tests
python classifier.py                       # snippets.csv -> predictions.csv
python evaluate.py                         # predictions.csv vs labels.csv -> report
python stability.py --runs 5               # optional: run-to-run variance (costs API calls)
```

The LLM escalation path needs a Groq key, read from a `.env` file in the repo root (or from
the real environment, which takes precedence):

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b     # optional, this is the default
```

Copy `.env.example` to `.env` to set it up. `.env` is gitignored and is not part of the
submission.

Useful flags: `python classifier.py --no-llm` forces the offline path;
`--in/--out` and `--truth/--pred/--snippets/--report` override paths.

**The committed results use the live LLM path**: 28 snippets decided by rules, 12 by
`openai/gpt-oss-120b`, 0 falling back. Without a key the pipeline still runs end to end —
escalated items go to a crude keyword scorer instead, always at LOW confidence — so the repo
is reproducible either way. The `route` column of `predictions.csv` records which stage
decided each row.

### Three things that had to be fixed to get the API working

Worth recording, since none of them are in the obvious place:

1. **Cloudflare blocks `Python-urllib`.** Groq sits behind Cloudflare, which rejects the
   default urllib user agent with `HTTP 403, Cloudflare error 1010`. Any non-default
   `User-Agent` header is accepted.
2. **Model availability is per-account.** `llama-3.3-70b-versatile` (my first choice) returns
   404 `model_not_found` on this key. `GET /openai/v1/models` lists what an account can
   actually reach; `openai/gpt-oss-120b` was the strongest available.
3. **Free tier is 8,000 tokens/minute.** Twelve calls at ~700 tokens each exceeds it, so the
   client retries on 429, preferring the server's `retry-after` header over guessing and
   falling back to exponential backoff. `gpt-oss-120b` is also a reasoning model — it spends
   completion tokens thinking before answering (~157 reasoning tokens/call), which is why
   `max_tokens` is 800 rather than the ~50 the JSON output needs.

Diagnosing (1) took longer than it should have because the first version of the client caught
every exception and returned a bare `None`. It now records the cause on `last_error` and the
classifier prints it. That is the one bug in this repo that cost me real time, and the fix is
a small argument for never swallowing an exception without recording why.

## How it works

Two stages, mirroring a human triage desk:

1. **Fast path** (`fast_path`) — five ordered lexical rules. They are ordered by veto strength:
   an explicit "nothing happened" statement beats a non-physical cause, which beats future
   framing, which beats topic, and only then does the positive event rule run. A rule returns
   only when it fires cleanly, with HIGH confidence; **any uncertainty hedge in the text blocks
   the positive rule outright**, because hedged cases are exactly what stage 2 is for.
   Negation is handled by a 3-token lookback (`_is_negated`), so "no oil spill has been
   reported" does not count as spill evidence.
2. **Escalation** (`llm_client.GroqClient`) — everything stage 1 refuses goes to an LLM at
   `temperature=0` with the rubric compressed into the system prompt, returning
   `{label, confidence, rationale}` as JSON. On a network or parse failure it degrades to the
   fallback scorer rather than crashing the batch.

On this data the split is 28 by rule, 12 escalated — and the 12 include almost every item I
myself marked LOW confidence, which is the main property I wanted. The LLM independently
returned LOW confidence on six of the eight items I had marked LOW, without ever seeing my
labels.

## Results in one line

Recall (YES) **0.944**, precision (YES) 0.895, accuracy 37/40. Three errors: id 18 (missed
event), ids 09 and 21 (false positives). Full numbers and the three written answers are in
[EVALUATION.md](EVALUATION.md).

Two findings I would lead with ahead of the score:

**Only one of the three errors is the classifier's fault.** The other two (Cases 2 and 3)
are defects in *my rubric* — a contradiction it never resolved, and two different notions of
"unconfirmed" collapsed into one rule. The classifier exposed both by applying the rubric more
literally than I ever did. Writing a spec precisely enough for a machine is a good way to find
out you had not written it precisely enough for a human.

**A 120B model and a word-counter score identically here.** Running the same pipeline with the
crude offline keyword scorer instead of `gpt-oss-120b` gives the same accuracy, recall and
precision, with the *same three items wrong* — the model was just more confident about one of
them. A 40-item, roughly balanced set cannot tell the two apart, which is the strongest
available argument that the headline metric is not yet measuring anything I would act on.

## Key decisions and trade-offs

**The classifier's confidence means something different from mine, on purpose.** In
`labels.csv`, LOW means "I had to make a judgement call". In `predictions.csv`, HIGH means "a
rule matched with nothing contradicting it". Keeping them distinct is what makes the
calibration table in `EVALUATION.md` informative rather than tautological — and it is why I
did not copy my confidences into the prompt.

**Rules first, LLM second — for cost, not accuracy.** At 100,000 items/day, sending everything
to an LLM is the wrong economics. The fast path is free and deterministic, so the LLM budget
is spent only on genuine ambiguity. The trade is real and I have named it: the fast-path NO
rules are unreviewable false-negative generators, and one of them lost snippet 18. See
`EVALUATION.md` §c for what I would change at volume.

**No scikit-learn, no pandas.** 40 rows and four metric definitions of four lines each. Adding
a dependency would have obscured the one decision that actually matters — that YES is the
positive class — behind a library default. Hand-rolling made it explicit and kept the repo
runnable with a bare interpreter.

**I fixed the specification, not the classifier — twice.** Both are recorded in `RUBRIC.md`:

- **Precedence** (from snippet 21): the LLM faithfully applied one rubric rule that
  contradicted another, and my rubric was silent on which wins. Writing the tie-break also
  forced out the distinction that keeps snippet 18 correct — a statement that *no incident
  occurred* is not the same as a benign outcome, and the sloppy phrasing would have broken the
  more important of my errors.
- **Corroboration** (from snippet 09): "unconfirmed" was one rubric row covering two different
  situations — an uncorroborated *event* versus corroborated event with unverified *numbers*.
  Splitting them makes 09 a NO while leaving 29 and 39 as YES.

But the **LLM system prompt is deliberately frozen** at the pre-fix rubric, so neither rule
reaches the classifier and the reported numbers stay at 37/40. Copying them into the prompt
would very likely fix both items — bought by editing the prompt after seeing exactly which
items failed, on the only 40 examples I have. That is tuning on the test set, and a classifier
tuned that way fails on the next dataset. Same reasoning for the two one-line rule-path fixes I
did not make. The line I am drawing: **fixing the specification is legitimate; fixing the
classifier against the specific items it failed is not.** Both rules propagate to the prompt on
the next run against held-out data (`EVALUATION.md` Cases 2 and 3).

**I checked determinism instead of assuming it.** `temperature=0` makes sampling greedy but
does not guarantee reproducibility. Across five runs the labels were identical every time —
so the headline metrics can be quoted without hedging — but the confidence on snippet 21
flickered between HIGH and LOW. That the model's instability landed on the one item my rubric
is self-contradictory about is the most interesting single result in the exercise.

**Stdlib `urllib` instead of the Groq SDK or `requests`.** One HTTP POST with a JSON body. A
dependency to save six lines was not worth making the repo un-runnable without a `pip install`.

### Where the time went

About **5 hours**, inside the 4–6 hour budget.

| Time | What |
|---|---|
| ~45 min | Reading the brief, drafting the rubric, deciding the awkward-case rulings before writing any code |
| ~60 min | Labelling all 40 snippets by hand, then a second pass over the boundary cases |
| ~75 min | Classifier and Groq client |
| ~45 min | Evaluation tooling and the 11 tests |
| ~30 min | Debugging the API — Cloudflare 403, model availability, the rate limit |
| ~55 min | Reading the results, the three disagreement cases, and the write-ups |

Two things are worth noting about that split. **The rubric came first, before any code**, so the
labels are an application of stated rules rather than a rationalisation of what a classifier
happened to produce. And **the evaluation and write-up took roughly as long as the classifier
did**, which matches where the brief says the marks are: "we are not scoring model
sophistication, we are scoring what you do next."

The 30 minutes of API debugging was unplanned and came out of the time I would otherwise have
spent on prompt iteration — which is why the system prompt is a single un-tuned draft.

### What I cut, and why

- **No embeddings / fine-tuning.** The brief says sophistication is not scored, and with 40
  labels there is nothing to fine-tune on. Time went into the evaluation instead.
- **No prompt iteration.** The system prompt is a single hand-written compression of the
  rubric. It has had zero tuning passes — the numbers below are what the first draft produced.
- **No batching or prompt caching.** Each snippet is one call carrying the full ~520-token
  system prompt. Fine for 40 items, wrong for 100,000/day; noted as the first cost fix in
  `EVALUATION.md` §c.
- **Only one model tried.** I did not compare `gpt-oss-120b` against the other models this key
  can reach. Given the evaluation set cannot distinguish a 120B model from a word-counter
  (see below), a model bake-off would have measured noise.
- **No cross-validation or bootstrap CI in code.** I computed the interval by hand in
  `EVALUATION.md` to make the point about sample size; automating it on n=40 was not worth it.

## What I would do with two more weeks

1. **Fix the ground truth first, not the model.** Everything else is capped by it. Two more
   annotators on the same sample, Cohen's κ to measure agreement, and disagreements adjudicated
   into worked examples in the rubric. Where agreement is low the rubric is underspecified —
   that is a spec bug, and it is the highest-value thing to fix.
2. **Get a bigger, realistically-skewed evaluation set.** 300–500 items sampled at the true base
   rate, held out and never used for tuning. Precision measured at ~45% prevalence (as here)
   says almost nothing about precision at 1%.
3. **Replace the binary with a calibrated score, then add tiers.** Choose the operating point on
   a precision-recall curve against a recall floor, and route by score into auto-drop / analyst
   queue / alert. This is the biggest structural limitation of the current build: there is no
   threshold to tune. Cyber/BI belongs here too, as a third class set by client policy rather
   than hard-coded as a NO inside a rule (`EVALUATION.md` §a, fourth disagreement).
4. **Enrich beyond the binary**: peril type, geography, severity estimate, and entity linking to
   the exposure book. "There is a fire in Rotterdam" is far more useful when it also says
   whether we insure anything within 2km of it. This is where the commercial value actually is,
   and none of it is in scope for a YES/NO classifier.

## AI usage

I used Claude Code throughout. Honestly:

**What the AI did.** Wrote most of the Python — the dataclasses, the phrase-matching and
negation helpers, the Groq HTTP client, the `.env` loader, the retry/backoff logic, the metric
functions, the report formatter, `stability.py`, and the test file. Transcribed the 40
snippets from the PDF into `snippets.csv`. Diagnosed the three API problems (Cloudflare 1010,
model availability, the 8k TPM limit) by probing the endpoint. Drafted the prose structure of
all four markdown files. Produced a first-pass set of 40 labels by applying my draft rubric,
and a first draft of the three disagreement analyses.

**What I decided.** The rubric is mine — the YES/NO boundary and, in particular, every
awkward-case ruling in it: unconfirmed reports as YES/LOW, forecasts as NO, precautionary
responses as YES, cyber as out-of-scope-but-flagged, drought as YES only when a consequence is
already in force. Those came from thinking about what an insurer actually needs to see, before
any code existed. Every one of the 40 labels in `labels.csv` I reviewed and own; the AI's
draft was a starting point I checked line by line, and `labels.csv` was never generated from
classifier output.

Mine also: the two-stage architecture and where the escalation boundary sits; the decision to
keep the classifier's confidence semantically distinct from mine; the choice of recall as the
headline metric and the argument for it; **the decision not to fix snippets 18 and 21**, which
is the judgement I would most want to be asked about; the "right answer, wrong reason" audit,
which came from reading the `route` column against the snippets by hand; the call to run the
offline scorer and the 120B model as a controlled comparison, which produced the finding I
think matters most; the decision to measure determinism rather than assume it; the reading of
Cases 2 and 3 as *rubric* defects rather than model errors, the wording of both rules that fix
them, and the decision to freeze the prompt rather than cash those defects in for a better
score.

The snippet 09 revision is mine and it is the one I would most expect to be challenged. I
labelled it YES on the first pass, straight off my own unconfirmed-report rule, and changed it
to NO on re-reading — after predictions had been generated. I have argued in `EVALUATION.md`
Case 3 that it is a genuine correction (it *cost* me accuracy and precision, and the rule it
produced generalises to 29 and 39 without special-casing), and I have also written down why
that argument is not fully sufficient and what the procedural fix would be. I would rather
surface that than quietly re-run the numbers.

I have read and understood every line in this repo. The places I would point to first if
challenged are `_is_negated` (a 3-token lookback, deliberately crude, and it is what makes
snippets 03 and 18 behave differently), the rule ordering in `fast_path` (the vetoes are
ordered by strength and reordering them changes answers), `_retry_delay` in `llm_client.py`
(prefers the server's `retry-after` over guessing), and `prf` in `evaluate.py` (which class is
positive is the whole argument).

## Citations

- Groq API request/response shape follows the OpenAI-compatible chat-completions endpoint
  documented at <https://console.groq.com/docs/api-reference>. The client code is written from
  scratch against that spec, not copied. The available-model list came from
  `GET /openai/v1/models` on my own key; the `retry-after` header behaviour on 429 is Groq's
  documented rate-limit response.
- Cloudflare error 1010 ("browser signature banned") is what surfaces when the default
  `Python-urllib` user agent is used against the Groq endpoint — diagnosed by probing, not
  from documentation.
- Precision/recall/F1 and macro-averaging follow the standard definitions (as in
  `sklearn.metrics`); implemented here directly rather than imported.
- Wilson score interval used for the 83–99% confidence interval quoted in `EVALUATION.md`.
- Snippet text is transcribed verbatim from `Leadenhall Candidate Assessment.pdf`.
- No other external code, datasets, or snippets were used.
