# Labelling rubric — insurable loss events

**Scope:** P&C / marine / energy, near-real-time. One question: *should an analyst look at this
now, because a physical event that could produce a claim has just happened?*

## Core rule

**YES** — a specific physical event, **already begun or just occurred**, with at least one of:
damage or destruction; injury, death, or people missing; or an active emergency or precautionary
response (evacuation, rescue, salvage, emergency shutdown, tow, emergency landing).

**NO** — everything else, especially anything whose subject is **money, words, or the future**:
markets, corporate news, funding, labour disputes, regulation, litigation, research, forecasts,
sport. Insurance vocabulary alone (reinsurance pricing, premiums, insurtech) is not evidence of an
event — the strongest false-positive trap in this data.

**Test:** (1) specific event, specific place, already begun? (2) physical? (3) damage, injury, or
active response? All three → YES.

**Confidence:** `HIGH` = no realistic alternative reading. `LOW` = a judgement call. Confidence is
my certainty in the *label*, not the severity of the event. Every LOW carries a justification.

## Awkward cases

- **Corroborated event, unverified details** → **YES, LOW**. Developing situations count.
- **Event category rests on one observer**, best-placed party silent → **NO, LOW**.
- **Forecast / warning / seasonal outlook** → **NO, HIGH**. Nothing has happened yet.
- **Precautionary response, damage unconfirmed** → **YES, LOW**. The response is itself the
  evidence: pipeline shut, dam evacuated, ferry towed, emergency landing.
- **Routine or explicitly no-incident** → **NO, HIGH**. No damage, injury, or response.
- **Cyber / BI with no physical damage** → **NO, LOW**. Right loss, wrong line of business.
- **Legal or financial consequence of a past event** → **NO, HIGH**. Already reserved.
- **Long-onset condition** (drought) → **YES, LOW**, only if a consequence is already in force.
- **Event confirmed, loss not yet quantified** → **YES, HIGH**. Loss size comes later.
- **Cause unknown, damage visible** → **YES, HIGH**. Cause matters to subrogation, not occurrence.

## Precedence

**An explicit statement that *no incident occurred* — not merely that no one was hurt — plus a
stated return to normal, beats the precautionary rule.** Snowfall closing passes with "no
incidents" and roads "reopened by morning" is NO; an emergency landing "with no injuries" is YES,
because the diversion *is* the event. Unresolved situations stay YES.

**Sourcing is judged on the event, not the numbers**: did the event *category* need a judgement
call? Floods from multiple sources → YES; a "partial collapse" from one observer → NO.

**Known bias:** recall-biased towards YES; cost trade argued in `EVALUATION.md` §c.
