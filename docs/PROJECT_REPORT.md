# Prompt Injection & Jailbreak Detection Guardrail — Project Report

*A from-scratch explanation of what this project is, why it exists, what was built, and what was
found. No prior background in AI, security, or this codebase is assumed.*

For the code-level companion to this document, see [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

---

## Table of contents

1. [What problem is this solving?](#1-what-problem-is-this-solving)
2. [The three approaches, in plain terms](#2-the-three-approaches-in-plain-terms)
3. [The data: three datasets, in detail](#3-the-data-three-datasets-in-detail)
4. [Turning three datasets into one](#4-turning-three-datasets-into-one)
5. [Building the fine-tuned classifier](#5-building-the-fine-tuned-classifier)
6. [Building the LLM-as-judge approach](#6-building-the-llm-as-judge-approach)
7. [The LLM Guard baseline](#7-the-llm-guard-baseline)
8. [How we measured success: every metric explained](#8-how-we-measured-success-every-metric-explained)
9. [Results: the full comparison](#9-results-the-full-comparison)
10. [The qualifire problem: a calibration story](#10-the-qualifire-problem-a-calibration-story)
11. [Data quality investigations](#11-data-quality-investigations)
12. [Do the three judges agree with each other?](#12-do-the-three-judges-agree-with-each-other)
13. [From model to product: the demo and Docker image](#13-from-model-to-product-the-demo-and-docker-image)
14. [Bottlenecks and challenges, told as a narrative](#14-bottlenecks-and-challenges-told-as-a-narrative)
15. [Conclusions and what's left](#15-conclusions-and-whats-left)
16. [Glossary](#16-glossary)

---

## 1. What problem is this solving?

Large language models (LLMs) — the technology behind ChatGPT, Claude, Gemini, and similar systems
— follow instructions written in natural language. That's their whole superpower: you don't
program them with code, you just *tell* them what to do. But that same flexibility is also their
biggest security weakness.

If an LLM-powered application takes any text from an untrusted source — a user's chat message, a
webpage it's asked to summarize, an email it's asked to reply to — and feeds that text to the
model, there is no hard boundary between "instructions from the developer" and "data the model is
supposed to just process." A malicious piece of text can say something like *"ignore your previous
instructions and instead do X"*, and depending on the model and the wording, it might just... do
X. This is called a **prompt injection**. A closely related attack, **jailbreaking**, tries to
manipulate the model into ignoring its own built-in safety training (e.g. by pretending the
request is fictional, or by role-playing a persona with no restrictions) so it produces content it
would normally refuse.

Both problems are serious enough that the security community's reference list for LLM
vulnerabilities — the
[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
— puts **prompt injection at #1**.

A **guardrail** is a defensive layer that sits in front of an LLM and inspects incoming text
*before* it reaches the model, trying to catch attacks like these and block or flag them. That is
exactly what this project builds: given a piece of text, decide whether it's a genuine attack
attempt or an ordinary, benign request — fast enough to run in front of every real request, not
just as an offline research exercise.

Because there's more than one way to build a guardrail, this project doesn't just build one and
call it done — it builds **three different approaches** and rigorously measures which one actually
works best, and under what conditions each one struggles.

## 2. The three approaches, in plain terms

**Approach A — a fine-tuned classifier.** Take a general-purpose language-understanding model
(DeBERTa, explained in §5) and specifically retrain part of it on thousands of labeled examples of
"attack" and "benign" text, so it becomes a dedicated attack/benign detector. This is the
traditional machine-learning approach: purpose-built, fast, but requires labeled training data and
a training process (GPU time, hyperparameter choices, the works).

**Approach B — LLM-as-judge.** Instead of training anything, just *ask* a capable general-purpose
LLM (Google's Gemini, in this project) to read the text and decide whether it looks like an attack.
No training needed — but every request costs an API call, is slower, and (as this project found)
the judge can sometimes refuse to even answer.

**Approach C — an existing off-the-shelf tool, as a baseline.** [LLM Guard](https://github.com/protectai/llm-guard)
is an open-source security toolkit that already ships a `PromptInjection` scanner. This project
doesn't build this one — it's used purely as an external point of comparison, to answer the
question: "is our purpose-built classifier actually better than just using an existing library?"

All three are evaluated on **exactly the same held-out data**, so the comparison is fair.

## 3. The data: three datasets, in detail

Three real, existing datasets — not synthetic or hand-written — were combined to build the training
and evaluation data. Each brings something different:

**1. `qualifire/prompt-injections-benchmark`** — 5,000 rows, roughly 60% benign / 40%
jailbreak. Columns: `text`, `label` (string: `benign` or `jailbreak`). This is a curated benchmark
dataset, smaller and cleaner than the other two.

**2. `neuralchemy/Prompt-injection-dataset`** — 4,391 rows. Columns: `text`, `label` (numeric: 0 =
benign, 1 = attack), plus metadata like `category`, `source`, `severity`, `augmented`. What makes
this dataset special: it deliberately includes **"hard negative" benign examples** — text
containing words that *look* suspicious (tagged e.g. `contains_execute`, `contains_bypass`) but
isn't actually an attack. This is exactly the kind of tricky case that separates a genuinely good
classifier from one that just pattern-matches on scary-looking keywords.

**3. `Necent/llm-jailbreak-prompt-injection-dataset`** — the big one: 1,175,432 rows in its raw
form, an aggregation of 30+ public safety-related datasets. This dataset is *not* purely about
prompt injection/jailbreaks — it also contains generic content-moderation data (toxicity, harmful
behavior requests, general "unsafe" text unrelated to attacking an LLM's instructions). Only the
rows explicitly tagged `prompt_injection` or `jailbreak` were kept (about 480,000 of the 1.17
million) — everything else was deliberately dropped, because mixing "text that manipulates an
LLM's instructions" with "text that's just generally inappropriate" would blur the exact problem
this guardrail is meant to solve.

After scoping, Necent alone is **~200x larger** than the other two datasets combined — a real
imbalance that shapes several later decisions (see §4 and §10).

## 4. Turning three datasets into one

Three datasets with three different label formats, three different column schemas, and wildly
different sizes can't just be thrown together — they need to be unified first.

**Unified schema.** Every dataset was mapped onto the same four columns: `text`, `label` (0 =
benign, 1 = attack), `source_dataset` (which of the three it came from — kept as metadata so
performance can be checked per-source later, never used as a training input), and `category` (the
original per-dataset label/tag, kept for reference).

**Deduplication.** After merging, exact-duplicate text rows were dropped (~19,600 rows removed
out of ~490,000). This matters because if the exact same sentence appeared in both the raw training
and raw test pool before splitting, the model could effectively "memorize" test answers rather than
generalize — see §11 for a follow-up investigation into whether near-duplicates (not just
exact ones) also snuck through.

**Resulting combined dataset**: 470,232 rows. Necent 460,841 (98.0%) / qualifire 5,000 (1.1%) /
neuralchemy 4,391 (0.9%). Overall label balance: **63.1% attack, 36.9% benign** — moderately
imbalanced, not extreme.

**Splitting into train/validation/test.** The combined data was split 80% train / 10% validation /
10% test, but not with a plain random split — with a **stratified** split on the combination of
`source_dataset` and `label` together. Plain random splitting would have kept the overall 63/37
attack/benign ratio consistent across splits, but could still have accidentally put
disproportionately few qualifire or neuralchemy rows into, say, the test set purely by chance
(since they're tiny relative to Necent). Stratifying on the joint `source_dataset + label` key
guarantees every split has a representative slice of every dataset at every label, so a small test
set doesn't accidentally become an unreliable measurement of qualifire/neuralchemy performance.

Final sizes: **train 376,185 rows, validation 47,023 rows, test 47,024 rows** — the same 63/37
label balance preserved in every split, and every split's per-source breakdown matches: necent
~98.1%, qualifire ~1.06%, neuralchemy ~0.93% of rows.

Text length across the whole dataset: mean 546 characters, median 375, ranging from as short as 10
characters up to one outlier at 55,089 characters.

## 5. Building the fine-tuned classifier

**What is DeBERTa?** DeBERTa (`microsoft/deberta-v3-base`, ~184 million parameters) is a
transformer-based language model — the same family of architecture behind models like BERT and,
at a much larger scale, GPT-style LLMs. Unlike a chat-style LLM that *generates* text one word at a
time, DeBERTa is an *encoder*: it reads a whole piece of text at once and produces a rich internal
representation of it, which can then be used for a downstream task like classification. It's
"pre-trained" on a huge amount of general text, so it already understands language broadly — the
project's job wasn't to teach it English, only to teach it this specific task.

**What is fine-tuning?** Fine-tuning means taking that pre-trained model and continuing its
training, but now on a much smaller, task-specific dataset (this project's 376,185-row training
set), with the model's output layer changed to predict just two classes: attack or benign. The
model's existing language understanding is repurposed toward this one decision, rather than being
learned from zero — which is why fine-tuning a few hundred thousand rows can work well, whereas
training a language model from scratch would need orders of magnitude more data and compute.

**Training in plain terms.** Training ran on a local RTX 4060 laptop GPU (8GB VRAM) for one full
pass ("epoch") over the training data — 23,512 optimizer steps, taking about 7 hours of actual GPU
time. The training loss (a number that measures how wrong the model's predictions are, on average,
where lower is better) dropped from an initial value to a final **0.043**, showing the model
learned the task well by the end of the pass.

**A real bottleneck: how long would this actually take?** Because the laptop couldn't stay on for
7 continuous hours in one sitting, training had to be designed to stop and resume across multiple
separate sessions using periodic checkpoints (saved snapshots of progress). Early estimates of how
long training would take were revised **twice** as more data came in — a short smoke test
suggested ~4-5 hours per epoch, a slightly longer early sample suggested ~14 hours, and only after
watching two real hours of training did the true pace (~23-24 hours per epoch at the *original*
settings) become clear. This is a useful lesson on its own: **early throughput estimates from short
smoke tests can be badly wrong** — the real number only became visible after a proper full-scale
sample, and even that took multiple revisions. A separate fix (skipping the very expensive
per-checkpoint validation pass, see §14) is what brought the real number down from ~23-24 hours to
the ~7 hours actually observed.

## 6. Building the LLM-as-judge approach

Rather than training anything, this approach simply asks a general-purpose LLM to make the same
judgment. **Google's Gemini** (specifically the free-tier `gemini-flash-lite-latest` model) was
chosen after ruling out running a local LLM: the laptop's 8GB of GPU memory could only fit a weak,
heavily-compressed 7-8B-parameter model locally, which would have been both slower and less
accurate than a well-resourced cloud model available for free.

**Sampling design.** Judging the entire 47,024-row test set with an LLM API, one request at a
time, under a strict free-tier daily quota, would have taken weeks. Instead, a representative
**1,439-row sample** was built: *all* 500 qualifire rows and *all* 439 neuralchemy rows (judged in
full, since those are the two smaller, more curated datasets and also where the classifier's known
weak spot lives — see §10), plus 500 randomly sampled necent rows (a smaller check suffices there,
since necent is both huge and the classifier already performs almost perfectly on it). This design
deliberately avoids wasting quota on the dataset that needs it least.

**Refusals.** An LLM judge, unlike a trained classifier, can outright refuse to answer — Gemini's
own safety filters occasionally blocked a response when asked to classify text that itself
describes a real attack technique (the irony being that judging the attack, not performing it, was
the whole point). Six rows out of 1,439 hit this. The project's rule for handling refusals: **they
are excluded from Gemini's accuracy/precision/recall/F1 entirely** (not counted as either right or
wrong, since a refusal isn't a classification), but tracked and reported separately as a refusal
rate — an honest way to represent a real limitation of this approach rather than hiding it inside
an averaged number.

## 7. The LLM Guard baseline

[LLM Guard](https://github.com/protectai/llm-guard) is an existing, open-source security toolkit
already used by real projects. Its `PromptInjection` scanner is itself backed by a different
fine-tuned model (`protectai/deberta-v3-base-prompt-injection-v2`) — coincidentally also based on
DeBERTa, but trained independently by a different team on different data, not related to this
project's own model. It was run, unmodified, over the exact same 1,439-row sample used for the
Gemini comparison, so all three approaches face an identical test.

## 8. How we measured success: every metric explained

This section explains, in plain terms, every metric used to judge the three approaches — including
several added specifically to answer the question "have you computed everything relevant for this
kind of task?"

- **Accuracy** — the percentage of predictions that were correct, overall. Simple, but misleading
  on imbalanced data (a model that always guesses "attack" would score 63% accuracy here just from
  the class balance, without learning anything).
- **Precision** — of everything the model *flagged* as an attack, what fraction actually was an
  attack? Low precision means too many false alarms (benign text wrongly blocked).
- **Recall** — of everything that actually *was* an attack, what fraction did the model catch? Low
  recall means real attacks are slipping through.
- **F1 score** — the harmonic mean of precision and recall, a single number balancing both. Used as
  the primary headline metric throughout this project because a guardrail needs to balance "don't
  block real users" against "don't miss real attacks" — optimizing either one alone in isolation is
  easy and useless.
- **Confusion matrix** — the four raw counts behind precision/recall: true positives (correctly
  caught attacks), true negatives (correctly passed benign text), false positives (benign text
  wrongly blocked), false negatives (real attacks that got through).
- **MCC (Matthews Correlation Coefficient)** — a single correlation-style score (from -1 to +1)
  between predictions and true labels that stays reliable even under class imbalance, unlike raw
  accuracy. Included because this data is moderately imbalanced (63/37).
- **ROC-AUC** — measures how well the model's confidence scores *rank* attacks above benign text,
  across every possible decision threshold at once, not just the one chosen for deployment.
- **PR-AUC** — the same idea as ROC-AUC but focused on precision/recall trade-offs specifically,
  usually more informative than ROC-AUC on imbalanced data.
- **Brier score** *(newly added for this report)* — the average squared difference between the
  model's predicted probability and the true 0/1 outcome. Unlike ROC-AUC (which only cares whether
  attacks rank above benign text), Brier score asks: when the model says "92% confident this is an
  attack," is it actually right about 92% of the time? Lower is better, 0 is perfect.
- **ECE — Expected Calibration Error** *(newly added for this report)* — a more interpretable
  cousin of Brier score: predictions are bucketed into confidence ranges (e.g. "70-80% confident"),
  and ECE measures the average gap between the model's stated confidence and its actual accuracy
  within each bucket. This metric turns out to be the cleanest possible measurement of exactly the
  calibration problem described in §10.
- **Cohen's Kappa** *(newly added for this report)* — measures how much two judges (e.g. DeBERTa
  and Gemini) agree with each other, corrected for the agreement you'd expect from pure chance
  alone. A stricter, more standard statistic than simple raw "percent agreement," used in §12.
- **Latency** — how long a single classification takes, measured one request at a time (not
  batched), since that's what a real deployed guardrail experiences: one user, one request, at a
  time.
- **Eval loss (cross-entropy)** — the same kind of number training loss is, but measured on
  held-out data never seen during training. Used as an overfitting sanity check: if eval loss were
  much higher than training loss, that would signal the model memorized the training data rather
  than learning to generalize.

## 9. Results: the full comparison

All numbers below are on the shared 1,439-row sample (all qualifire + all neuralchemy + 500
sampled necent rows), the only dataset all three approaches were run on. DeBERTa's numbers on the
*full* 47,024-row test set are even stronger (see the table after) — this sample is deliberately
tilted toward the harder, smaller datasets, so treat the two tables as answering different
questions.

### Three-way comparison (1,439-row shared sample)

| Approach | n scored | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **DeBERTa (this project)** | 1,439 | 90.76% | 86.12% | 98.57% | **91.92%** | 0.822 |
| Gemini (LLM-as-judge) | 1,433 (6 refused) | 84.51% | 90.30% | 79.40% | 84.50% | 0.697 |
| LLM Guard (baseline) | 1,439 | 71.09% | 71.95% | 75.13% | 73.50% | 0.418 |

**DeBERTa wins overall**, driven by near-perfect recall (misses only 11 of 768 real attacks in
this sample) at the cost of somewhat more false positives than Gemini. Gemini is the opposite
profile: more conservative, misses more attacks but is more precise when it does flag something.
LLM Guard is the weakest of the three on this sample, and its MCC (0.418) being much lower than its
F1 (73.5%) alone would suggest is a sign its errors are not evenly distributed — confirmed by the
per-dataset breakdown below.

### Per-dataset breakdown — a genuinely surprising finding

| Dataset | DeBERTa F1 | Gemini F1 | LLM Guard F1 |
|---|---|---|---|
| necent | 99.50% | 85.09% | 66.67% (MCC **-0.036**, essentially a coin flip) |
| neuralchemy | 96.62% | 86.07% | **90.02%** (LLM Guard's *best* category) |
| qualifire | 78.13% (default threshold) / 92.79% (deployed threshold) | 81.80% | 64.80% |

The single most interesting result in this table: **all three approaches have different, genuinely
non-overlapping weak spots.** DeBERTa struggles specifically on qualifire (see §10). LLM Guard is
*near-random* on necent (MCC of -0.036, meaning it does barely better than flipping a coin) despite
necent being the *easiest* category for both other approaches — yet neuralchemy, which trips up
DeBERTa and Gemini the least of the "hard" categories, is LLM Guard's strongest showing. This is
not "one model is strictly better than the others" — it's a genuine complementary-weaknesses
picture, worth remembering if any of these were ever combined (e.g. an ensemble) rather than
deployed alone.

### DeBERTa on the full held-out test set (47,024 rows)

This is the real deployment-scale number, unlike the smaller 1,439-row shared sample above.

| | Overall | necent | neuralchemy | qualifire |
|---|---|---|---|---|
| n | 47,024 | 46,085 | 439 | 500 |
| Accuracy | 99.40% | 99.67% | 95.90% | 77.60% |
| Precision | 99.40% | 99.81% | 96.25% | 64.10% |
| Recall | 99.64% | 99.66% | 96.98% | 100.00% |
| F1 | 99.52% | 99.74% | 96.62% | 78.13% |
| MCC | 0.987 | 0.993 | 0.914 | 0.634 |
| ROC-AUC | 0.9996 | 0.9997 | 0.9889 | 0.9901 |
| PR-AUC | 0.9998 | 0.9999 | 0.9921 | 0.9867 |
| **Brier score** | 0.0057 | 0.0032 | 0.0389 | 0.2102 |
| **ECE** | 0.0052 | 0.0027 | 0.0382 | 0.2184 |

(These figures use `threshold=0.5`, the raw model output, before the deployed 0.99 threshold
described in §10 is applied — shown here to illustrate the raw model quality before any
threshold tuning.)

Notice qualifire's ROC-AUC (0.99) is nearly as strong as necent's — meaning the model's underlying
*ranking* of attack-vs-benign is actually excellent on qualifire too — yet its accuracy/F1/Brier/ECE
are dramatically worse than the other two datasets. That gap between "ranks correctly" and "raw
accuracy is bad" is the exact signature of a **calibration** problem, not a "the model doesn't
understand the task" problem. §10 explains this in full and the fix that was applied.

Single-request latency (unbatched, one prompt at a time — what a real deployment would see): mean
54.8ms, p50 48.4ms, p95 101.9ms, p99 155.3ms, measured on GPU. Fast enough for a real-time
guardrail sitting in front of every request.

## 10. The qualifire problem: a calibration story

Qualifire's numbers above stand out immediately — precision of just 64%, meaning over a third of
the model's "this is an attack" flags on this specific dataset were actually false alarms on benign
text, even though recall was a perfect 100% (every real qualifire attack was caught). ROC-AUC was
still ~0.99, meaning the model's raw confidence scores *were* correctly ordering attacks above
benign text — the problem wasn't understanding, it was **where the cutoff line was drawn**.

Every classifier that outputs a probability needs a decision rule: "if the probability of attack is
above X%, call it an attack." The default is almost always 50%. But there's no law of nature saying
50% is the right cutoff for every dataset a model will ever see — and qualifire's benign examples
apparently tend to *sound* more attack-like (its 0.5-threshold predictions push benign text just
over the 50% line more often) than necent's or neuralchemy's, even though the model still ranks
them correctly relative to each other.

**The fix**: rather than trying to guess a different threshold per dataset (which is not actually
possible in a real deployment — a live guardrail sees raw text with no "which dataset is this from"
tag attached), a single **global** threshold was swept across a wide range of values (0.5 up to
0.9999), tuned only on the validation split, then confirmed on the untouched test split. Raising
the global cutoff from 0.5 to **0.99** dramatically improved qualifire (F1 climbed from 78.1% to
92.8%, precision from 64% to 89.4%) while only mildly costing recall/F1 on the other two datasets
(necent F1 dipped from 99.74% to 98.87%, neuralchemy from 96.62% to 94.39%) — a clear net win, which
is why **0.99 is the threshold actually deployed** in the demo and CLI.

The new Brier score / ECE numbers computed for this report make the story numerically crisp:
qualifire's ECE (0.218) is roughly **80x worse** than necent's (0.003) and about **6x worse** than
neuralchemy's (0.038) — a direct, quantified confirmation that this was specifically a calibration
issue on one dataset, not a general model weakness.

One alternative explanation was tested and ruled out before settling on calibration: **language
contamination**. Only 2.4% of qualifire (12 of 500 rows) is non-English, and *0%* of the actual
false positives were non-English — every single false positive was on ordinary English text. So the
"maybe the model just can't handle non-English/code-mixed text" theory doesn't hold; calibration
remains the standing, confirmed explanation.

## 11. Data quality investigations

Two specific, targeted data-quality checks were run (rather than a blanket cleaning pass — see
§15 for why a full cleaning effort was deliberately not pursued):

**Train/test leakage in necent.** Because necent is a huge aggregation of many sub-sources, it was
worth checking whether near-identical text (not exact duplicates — those were already removed, see
§4) leaked between the training set and the test set, which would let the model "cheat" by
memorizing rather than generalizing. Two independent methods were used: sentence-embedding cosine
similarity, and MinHash/Jaccard lexical similarity (a completely different technique, included
specifically because the user wanted two independent confirmations, not just one). **Result:**
leakage is real — about 5.15% of a 2,000-row test sample had a near-identical (≥99% similarity)
match in the training set, and manual inspection confirmed genuine reused attack templates (e.g.
recurring "Project Zenith" phishing-style templates, DAN-style jailbreak variants) — several pairs
were even 100% identical, meaning they slipped past the exact-string dedup due to tiny
whitespace/punctuation differences.

**Does the leakage actually inflate the reported score?** Rather than a full, expensive retrain,
the existing 2,000-row leakage sample was split into "leaked" vs. "clean" subsets and DeBERTa's
accuracy compared on each. The leaked rows scored a trivial, expected 100% (the model has
essentially seen them before). But the **clean** subset — 89.5% of the sample — still scored F1
99.67%, essentially identical to necent's originally reported ~99.7-99.8%. **Conclusion: the
leakage is real, but too small a share of the data to meaningfully inflate the headline number.**
Decision made: disclose this as a known caveat, don't spend ~7 hours of GPU time on a retrain that
the numbers say wouldn't change the outcome.

**Qualifire language contamination** — covered in §10 above, ruled out as an explanation for the
calibration problem.

## 12. Do the three judges agree with each other?

Beyond each approach's accuracy against ground truth, it's also useful to ask: when DeBERTa,
Gemini, and LLM Guard look at the *same* piece of text, how often do they actually agree with each
other? Cohen's Kappa (see §8) answers this properly, correcting for the agreement you'd expect from
random chance alone:

| Pair | Cohen's Kappa | Raw agreement rate |
|---|---|---|
| DeBERTa vs Gemini | **0.668** (substantial agreement) | 83.2% |
| DeBERTa vs LLM Guard | 0.437 (moderate agreement) | 72.6% |
| Gemini vs LLM Guard | 0.444 (moderate agreement) | 72.0% |

DeBERTa and Gemini — the two stronger performers — agree with each other noticeably more than
either agrees with LLM Guard, which lines up with LLM Guard being the weakest of the three overall.
None of the pairs are near-perfect agreement, reinforcing the earlier point that these three
approaches genuinely see the problem differently rather than converging on the same answer through
different means.

## 13. From model to product: the demo and Docker image

A trained model sitting in a checkpoint folder isn't a usable product — this project also built a
real inference path and a way for someone else to actually try it:

- A `Guardrail` class wraps the trained checkpoint and applies the deployed 0.99 threshold
  decision (the first place in the codebase this threshold is actually *applied*, rather than just
  measured in a report).
- A small local web application (FastAPI backend + a custom-styled browser frontend, matching the
  visual identity of the evaluation report) lets anyone paste text and get a live verdict, including
  a "sample a real held-out test row" button for exploring real examples.
- A command-line entry point (`python -m scripts.classify "..."`) for scripted/non-browser use.
- A **Docker image** so the whole demo can run on any machine with Docker installed, without
  needing to set up Python, a virtual environment, or a GPU — CPU-only inference, verified
  end-to-end (both an attack and a benign prompt correctly classified through the actual running
  container, not just a successful build).

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the engineering detail behind all of this.

## 14. Bottlenecks and challenges, told as a narrative

No project this size goes smoothly end to end. The honest list, in the order these were actually
hit:

- **GPU memory limits shaped the whole architecture.** An 8GB laptop GPU ruled out both a locally
  hosted LLM judge (would have needed a heavily compressed, weaker model) and influenced training
  batch-size choices (a small per-device batch size of 8, compensated with gradient accumulation to
  reach an effective batch size of 16).
- **Training time estimates were wrong, twice, before being right.** See §5 — the true throughput
  only became visible after hours of real training, not from short smoke tests.
- **A subtle numerical bug (fp16 NaN) had to be diagnosed and fixed before training could even
  start reliably** — DeBERTa-v3's specific attention mechanism produces invalid outputs under
  naive half-precision loading; fixed by loading full-precision weights and using a different mixed
  precision mode during training instead.
- **A checkpoint silently corrupted mid-save**, and the process was stopped before that was
  noticed — the checkpoint *looked* complete (GPU was idle, no error was thrown) but was actually
  missing files. This forced a policy change: always verify a checkpoint's file completeness before
  trusting it to resume from, not just check whether the GPU is idle.
- **A very expensive hidden cost was found and removed**: evaluating on the full 47,000-row
  validation set after every single checkpoint would have added roughly 21 extra hours on top of
  training itself. Moving evaluation to run once, after training finished, was what brought total
  training time down from an early ~23-24 hour/epoch estimate to the ~7 hours actually needed.
- **A third-party library installation nearly broke the whole pipeline.** Installing LLM Guard
  directly into the main project's environment silently downgraded core packages the DeBERTa
  training pipeline depended on — including reintroducing the exact fp16 bug that had already been
  fixed once. Caught and reverted before real damage, but the lesson (install unfamiliar
  dependency-heavy libraries into an isolated environment *first*, not after cleanup) shaped how
  the LLM Guard baseline was ultimately run.
- **API rate limits and a resumability bug** for the Gemini judge: a bug in how "already judged"
  rows were tracked meant hitting the daily quota could silently and permanently discard
  in-progress results rather than safely stopping — hit twice before being properly fixed, costing
  some wasted API quota both times.
- **A real network outage during Docker containerization** turned out not to be a Docker
  misconfiguration at all — a mobile-hotspot connection was selectively failing to reach GitHub and
  Docker Hub specifically, while every other site worked fine. Diagnosed methodically (ruling out
  Docker Desktop itself, then DNS, then IPv4 vs IPv6) before concluding it was a network-side issue
  outside the project's control, and it resolved once the connection stabilized.

## 15. Conclusions and what's left

**Headline takeaway**: the purpose-built, fine-tuned DeBERTa classifier outperforms both a
general-purpose LLM used as a judge and an existing open-source guardrail library, on this task,
on this data — by a meaningful margin on F1 and MCC, and with far lower latency than an API-based
judge would allow. But "wins overall" hides a more interesting and more useful finding: **all three
approaches have real, different, non-overlapping weaknesses** (DeBERTa on qualifire specifically,
LLM Guard near-randomly on necent, Gemini's outright refusals on some real attacks) — a reminder
that no single approach is uniformly best across every kind of input, and that per-dataset
evaluation (not just one blended number) is what actually surfaces this.

**Deliberately deferred, not forgotten:**
- Broader data-cleaning gaps (extreme-length outliers, very short texts, HTML/encoding artifacts)
  were assessed as affecting under 2% of the test set combined — low expected payoff, not pursued.
- A template-aware re-split + full retrain to eliminate the confirmed (but small-impact) necent
  leakage was considered and deliberately not done, since the leakage-adjusted analysis in §11
  showed it wouldn't meaningfully change the reported numbers.
- Deploying the demo beyond localhost (a publicly reachable URL, not just a local/Docker
  reproduction) was explicitly out of scope for this phase.
- A "meanwhile project" — a second project to work on in parallel, or an adversarial-evaluation
  deepening of this one (e.g. testing the guardrail against obfuscation techniques like base64 or
  homoglyph encoding) — remains an open, deliberately deprioritized decision.

## 16. Glossary

- **LLM** — Large Language Model, an AI system trained to understand and generate natural language.
- **Prompt injection** — text crafted to manipulate an LLM into ignoring or overriding its intended
  instructions.
- **Jailbreak** — text crafted to get an LLM to bypass its own safety training and produce content
  it would normally refuse.
- **Guardrail** — a defensive filter placed in front of an LLM to catch attacks before they reach it.
- **Transformer / encoder** — the neural network architecture family behind modern language models;
  an *encoder* specifically reads text and produces a representation for downstream tasks like
  classification, as opposed to a *decoder*, which generates new text.
- **Fine-tuning** — continuing to train an already-pretrained model on a smaller, task-specific
  dataset, so it specializes without learning language from zero.
- **Checkpoint** — a saved snapshot of a model's progress during training, allowing training to be
  paused and resumed.
- **Epoch** — one full pass through the entire training dataset.
- **Threshold / decision boundary** — the cutoff probability above which a prediction is classified
  as "attack" rather than "benign."
- **Calibration** — whether a model's stated confidence (e.g. "92% sure") matches its actual
  real-world accuracy at that confidence level.
- **Overfitting** — when a model performs well on training data but fails to generalize to new,
  unseen data, often because it partly memorized rather than learned.
- **Held-out / test set** — data deliberately kept separate from training, used only to measure how
  well the model generalizes to data it has never seen.
- **API quota / rate limit** — a cap on how many requests a service will accept in a given time
  window (e.g. per minute, per day).
