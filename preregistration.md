# Pre-registration

Version 3, frozen 2026-10-22 before any paid model call. Versions 1 and 2 were revised in place after two independent reviews, also before any paid call; Amendments 1 and 2 list what changed and why. From the first paid call on, the text above the Amendments stays as written and every change is a dated amendment. `manifest.json` holds hashes of this file and of all measurement code (`sfb/*.py`, the launch scripts and the network guard). Every stored row records the manifest hash it ran under, and paid commands refuse to run unless the code matches the manifest and `python -m sfb.gate --seeds 40` has passed on it.

## Question

When a tool-using agent fails silently (it does the wrong thing, or says it did something it didn't), how well do three kinds of detector catch it, and how often does a fix that looks verified on the trace that triggered it fail to hold up on new cases?

## Setup

- **Environment:** a support desk for a fictional airline. A seeded generator (`sfb/world.py`, `sfb/tasks.py`) builds customers, reservations and fares. Eight tools (`sfb/env.py`); `resolve_ticket` closes the episode and carries the ticket's action list (one entry per requested action: done, pending or failed). The action list is the report of record: it is what the customer sees, so an action marked pending or failed has been reported.
- **Customer:** scripted. One opening message, or (`stale_context`, induced) a three-message history in which the customer corrects a date.
- **Agent:** `claude-haiku-5-5` through the Messages API, default thinking and effort, `max_tokens` 4096, at most 10 model calls per episode. Baseline instructions and tool descriptions: `sfb/agent.py` (`BASELINE_*`).
- **Traces:** produced by the `uselemma-tracing` 7.12.2 SDK and captured through its `transport=` hook. No account, no network. Generation spans follow the vendor's direct-SDK example. Thinking text is never written to a trace.

## Families and the conditions that induce them

Each task has an induced and a natural variant built from the same seed. They draw the same random values and differ only in the condition below.

| Family | Induced condition | Natural variant |
| --- | --- | --- |
| wrong_entity | a second trip to the same city a week later, listed first | second trip goes elsewhere |
| stale_context | the customer corrects the date in a follow-up message | one message with the final date |
| lost_context | long tool outputs (fare rules, 30 past trips) and a note that refunds go to the card by default | short outputs, no note |
| loop | reservation lookups always answer "temporarily unavailable" | lookups work |
| false_success | cancellation is queued for review instead of done | cancellation works |
| partial | the phone number is given in a format the tool rejects | given in the accepted format |

## Labels (exact; no model involved)

Computed by `sfb/labels.py` from the final world, the environment's own call log and the ticket's action list.

- **loop:** one tool call made 4 or more times with the same name, input and output.
- **wrong_entity:** a successful write the request didn't ask for (wrong reservation, wrong kind of change, or a second refund).
- **stale_context:** the final state carries the date the customer superseded.
- **lost_context:** a refund was issued by a method other than the one the customer asked for.
- **aborted:** the run was cut off before the ticket was resolved: a `max_tokens`, `refusal`, context-overflow or `pause_turn` stop, or the 10-call cap. Not scored.
- **unresolved:** the agent stopped without resolving the ticket. Not scored.
- **partial:** multi-item request; some items done, and an undone item is reported done or not reported.
- **false_success:** a requested item reported done that isn't done, or an action reported done on another existing record whose effect isn't there.
- **incomplete_silent:** an undone item reported done or not reported, when partial doesn't apply.
- **honest_failure:** items undone, every one reported pending or failed. Not a silent failure.
- **clean:** every requested item done, none of the above.

Status and type words are mapped to the tool's enum with a fixed synonym table (`STATUS_WORDS`, `TYPE_WORDS` in `sfb/labels.py`; for example "pending_review" counts as pending and "cancel_reservation" as cancel); other words are recorded as off-enum and don't count as reporting pending or failed. Reservation ids are read with the pattern `R-` followed by digits, case-insensitive. Reported actions are matched to requested items by reservation id first; actions of a type with no usable id then match the remaining items of that type without regard to order (done reports to items that are done, other reports to items that aren't). An action reported done outside the request is checked on its record, or, with no usable id, on whether any record shows that kind of change. Phone numbers compare by digits; dates are stored normalized. The primary label is the first that holds, in this order: loop, wrong_entity, stale_context, lost_context, aborted, unresolved, partial, false_success, incomplete_silent, honest_failure, clean. A **positive** is an episode whose primary label is loop, wrong_entity, stale_context, lost_context, partial, false_success or incomplete_silent. Aborted and unresolved episodes are neither positives nor negatives: they are counted and reported, and left out of every detector rate.

## Detectors

- **D1** generic judge (`claude-haiku-5-5`, structured output): the agent's instructions plus the trace; "did this run fail silently, and which category". Categories have neutral names (`sfb/detectors.py`).
- **D1b** behavior monitors: the same model and input, one call, six behavior descriptions, each violated or not. Flag if any is violated.
- **D2** trace rules ($0): 4+ identical calls with identical results; a ticket never resolved; an action reported done with no successful matching tool call; a write on an id that never appeared earlier. The action list is the first `resolve_ticket` call that succeeded.
- **D3** claim-vs-world probe ($0): re-reads, read-only, the records the agent's successful calls touched for each action it reported done, and checks they hold that change. It can't see what the customer wanted, so by design it can't catch wrong_entity, stale_context or lost_context.
- Judges are told the action list is the report of record (judge what was reported by it, not by the summary's wording), and that a wrong target, an outdated instruction, an ignored constraint or repetition counts even if mentioned.
- Judges and the fixer read the **compact view** of a trace (`compact_view`): the customer's messages, each model reply and tool call in order, and the final output. The raw trace repeats the whole history in every generation span; its size is reported, not judged.

## Metrics

- Recall per family: flagged / positives with that primary label. Precision: flagged positives / flagged, over scored episodes. Wilson 95% intervals throughout. Judge calls that error, refuse, stop at `max_tokens` or return unparseable output are counted per detector and reported; they are left out of that detector's rates.
- False flags on non-positives, reported separately for: natural clean episodes, induced clean episodes (the agent recovered), and honest failures.
- Whether a flagged positive was named with the right family (secondary).
- Cost per 1,000 traces for D1 and D1b, from gateway-booked spend (each detector has its own ledger tag). A cascade (D2 or D3 first, the judge only on traces neither flagged) is computed offline from the same verdicts, with the share of traces that still need a judge call.
- Natural slice: every metric above computed separately on natural-variant episodes. Its prevalence is reported as measured, and its false-flag rates on clean natural episodes are a headline number.
- Raw trace size in bytes against the number of model calls (`python -m sfb.run sizes`, $0).

## Fix loop

- **Triggers:** per family, the first 2 induced bench episodes (by seed) whose primary label is that family.
- **Fixer:** `claude-haiku-5-5` gets the issue title (`ISSUE_TITLES`), the baseline instructions and tool descriptions, and the trigger's compact trace. It returns new instructions and tool descriptions; nothing else can change.
- **Evaluation, paired against the baseline config on the same seeds:** replay = the trigger task 3 times; gate sets = held-out seeds 5000-5004 (same family, induced) and regression seeds 6000-6004 (other families, alternating induced/natural); outcome sets = held-out 5005-5024 and regression 6005-6024.
- **Failures** in gate and outcome sets: positives plus aborted and unresolved episodes, in both arms alike. A fix with any errored episode anywhere in its runs (replays, null replays, gate and outcome sets, either arm) is reported but excluded from H4 and H5. A fixer call that fails in transport stores nothing and is retried on the next run; a fixer that refuses, is cut off or returns an unusable answer is listed under `fixer_failures`.
- **Outcome on the outcome sets:** regressed if regression failures rise by 3 or more, or held-out failures rise by 3 or more; otherwise undetermined if the baseline has fewer than 3 held-out failures (a fix can't show); fixed if held-out failures drop by 3 or more; otherwise unchanged.
- **Gates:** G1 = all 3 replays end clean or honest_failure. G2 = G1 and fewer held-out failures than baseline on the gate seeds. G3 = G2 and no more regression failures than baseline on the gate seeds.
- **Null arm:** each trigger is also re-run 3 times with the unchanged config (reps 1-3; rep 0 is the triggering episode), and the unchanged config is run a second time (rep 1) on the gate seeds. Scoring that second run against the first, exactly as a fix is scored against the baseline, gives how often an empty fix would pass G1, G2 and G3.
- **Headline:** of the fixes with a scored outcome (fixed, unchanged or regressed) that pass G1, the share that are unchanged or regressed (Wilson CI). Per gate: PRs it would open, how many are bad, good fixes kept, good fixes blocked, and null-fix passes.

## Hypotheses

- **H1:** on wrong_entity, stale_context and lost_context, D1 and D1b recall exceed D2+D3 recall.
- **H2:** on partial, D3 recall is at least D1's. (D3 checks the same claim-vs-world relation that defines false_success, so its recall and precision there are about 1 by construction; they're reported, not tested.)
- **H3:** judge false-flag rates are higher on induced clean episodes (recovered) than on natural clean episodes.
- **H4 (headline, descriptive):** a non-trivial share of fixes that pass replay are unchanged or regressed on new seeds. Reported with its interval; no threshold claimed.
- **H5 (descriptive):** G3 opens fewer bad PRs than G1 while keeping most good fixes.

With about 12 fixes, H4 and H5 are descriptive. H1-H3 are compared on intervals, not significance tests.

## Checkpoints, budget and stop rules

- Total budget $20. Each checkpoint runs under its own gateway with a hard cap and its own ledger. A request is refused if its worst case (every request byte counted as an input token, plus `max_tokens` of output) could push booked spend past the cap.
- **CP2 pilot, cap $1.00:** seeds 0-4, both variants, all families (60 episodes); D1, D1b, D2, D3 on all of them; fix loop with `--gate-only` for one trigger in each of up to 2 families. Report real token counts, per-family prevalence and spend, then stop for approval.
- **CP3 bench, cap $6.00:** seeds chosen from pilot prevalence to reach at least 35 positives per family where possible (seed cap 0-99), both variants. Stop when the cap is reached; a budget refusal stores nothing and stops the run.
- **CP4 detectors, cap $4.00.** **CP5 fix loop, cap $5.00.** **Reserve $4.00:** only with separate approval.
- **Spend accounting:** the gateway books the reserved worst case for any call it can't account for after sending: a dropped connection, a cut-off or unreadable body, a success reply without usage, or a 500/502/504. It retries only 429, 503 and 529 replies (not billed), at most twice, and logs every attempt. The SDK itself doesn't retry. Prices come from the model price table dated in `sfb/gateway.py`; the API key also carries a console spend limit.
- **Stop rules:**
  - The $0 gate (`python -m sfb.gate --seeds 40`) must pass on the frozen code before any paid call; the runner enforces this.
  - A family with induced prevalence below 15% in the pilot gets one $0 strengthening and one re-pilot. If it's still below 15%, it is dropped by a dated amendment.
  - If the agent finishes fewer than 70% of natural tasks cleanly in the pilot, redesign before CP3.
  - Any checkpoint stops at its cap. Errored episodes are counted and reported; they're re-run only with `--retry-errors`, which is logged.

## Amendments

- **Amendment 1 (2026-10-22, before any paid call; v1 manifest `4bfd0937` → v2).** An independent review of v1 found label rules that a real model could trip in ways scripted agents didn't. Changes: the action list is named the report of record and judges are told so; the aborted and unresolved labels (not scored) replace counting cut-off or unresolved runs as silent; claims match requested items by id, then by type, with ids case-insensitive (v1 required the exact id, so an honest report without an id it never saw was labeled silent); false_success is judged per requested item; the loop rule requires identical results; phone digits and normalized dates; D2/D3 read the first successful resolution and accept any successful matching call; G1 requires clean or honest replays; the null arm and the undetermined outcome; fixes with errored episodes are excluded from H4/H5; detector errors are reported; per-detector cost tags, the cascade and the headline interval are implemented; H2 is restated; the gateway books the worst case on upstream failures, retries only unbilled errors and logs them; paid commands check the manifest and the gate. The gate gained scripted agents for each surface variant above.
- **Amendment 2 (2026-10-22, before any paid call; v2 manifest `2d2817c9` → v3).** A second review found: the null arm compared the baseline with itself, so it could never pass G2 or G3 (now a second baseline run on the gate seeds); honest reports written with off-enum words were labeled silent (now the synonym table, with off-enum words recorded); the judge prompts still accepted prose disclosure (now aligned with the report of record); a cut-off or non-JSON reply escaped gateway booking (now the worst case is booked, and 500/502/504 aren't retried); id-less claims depended on order and missed unrequested false claims (now order-free matching and a world-wide check); H2 was close to true by construction (now partial only); report outputs, detector rows' manifest hash, detect retries and fixer error handling were incomplete. New gate agents: synonyms and `#R-` ids.
