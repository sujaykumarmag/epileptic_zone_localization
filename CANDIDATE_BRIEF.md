# Technical Assessment — Node-Level Abnormality Localisation on Multimodal Graphs

**Time limit:** 24 hours from receipt. **Designed effort:** ~11 hours.
**Submission:** a single git repo or `.zip` (notebook + code + report + `predictions.csv`).

> **Please read this first.** The REQUIRED sections are what we grade, and they are
> sized to fit a normal working day. STRETCH items are genuinely optional. A
> careful, honest, complete REQUIRED submission scores **higher** than a rushed
> attempt at everything. Please do not pull an all-nighter — we are not measuring
> stamina.

>
> `check_data.py` verifies your arrays are correctly aligned and your submission
> is valid. It does **not** load data for you — writing that is part of the task,
> and a silent alignment bug would invalidate everything downstream, so run the
> check early.

---

## 1. Background (no domain knowledge needed)

You have data for **140 independent subjects**. Each subject is a **graph of 68
nodes**. Nodes are interconnected entities; edges are weighted and represent
connection strength. The node set and its ordering (`node_id` 0–67) are
**identical across all subjects** — only edge weights and node features differ.

In every subject a **small subset of nodes (3–7, about 7%) is "abnormal."** Your
job is to find them.

Each node carries features from **two measurement modalities**: `modality_A`
(6 features) and `modality_B` (7 features).

You also get, in a separate file, an **independently generated simulation score**
per node, produced by a different pipeline that does not use the node features at
all. It comes with a per-subject confidence value.

No neuroscience, biology or medical background is required, expected, or graded.

---

## 2. Your task

Output, for every node of every subject, a **probability that the node is
abnormal** — and answer the four questions in Section 5.

---

## 3. The data

Everything is in `data/`. All tables key on `(subject_id, node_id)`.

| File | Rows | Contents |
|---|---|---|
| `subjects.csv` | 140 | `subject_id`, `split`, `site`, `hemisphere`, `n_nodes`, `modality_B_available`, `engel_1_seizure_free` |
| `modality_A.csv` | 9,520 | 6 features: `ct_z`, `gmv_z`, `sa_z`, `curv_z`, `t1t2_z`, `lgi_z` |
| `modality_B.csv` | 7,140 | 7 features: `rel_delta`, `rel_theta`, `rel_alpha`, `rel_beta`, `rel_gamma`, `spike_rate`, `hfo_rate` |
| `simulation_scores.csv` | 9,520 | `sim_score`, `sim_confidence` |
| `node_labels.csv` | 9,520 | `is_abnormal` (**target**), `resected` |
| `region_names.csv` | 68 | `node_id` → `region` |
| `adjacency/<sid>_adj.npy` | 140 files | float32 `68 × 68` weighted adjacency |

**Adjacency:** symmetric, non-negative, zero-diagonal, ~30% dense, **unnormalised**.
How you normalise is your choice.

**Missing data — two distinct kinds, both deliberate:**

1. **Whole modality missing.** 35 of 140 subjects (25%) have **no rows at all** in
   `modality_B.csv`. Flagged by `subjects.csv → modality_B_available`. Spread
   across all three splits.
2. **Scattered values missing.** ~6% of cells in `modality_A.csv` are `NaN`,
   missing at random.

**`sim_confidence`** is constant within a subject and varies between subjects. It
is a property of the simulation pipeline, **not** of the target.

**`resected`** and **`engel_1_seizure_free`** are for the optional Section 5.5
only. **They must never be used as model inputs or training targets.**

### Splits — use exactly these
`subjects.csv → split` gives `train` (84) / `val` (28) / `test` (28). The split is
**by subject**. Fit on `train`, select and tune on `val`, and touch `test` **only**
for your final reported numbers. Labels are provided for all splits — we are
trusting you, and we will read your code.

---

## 4. Metrics — report these on `test`

Classes are imbalanced (~7% positive), so **accuracy is meaningless**. Implement
these once (notebook section 2) and reuse them throughout.

| Metric | Notes |
|---|---|
| **AUPRC** | **primary.** Always report the prevalence baseline (~0.073) beside it |
| **AUROC** | pooled over test nodes |
| **Top-k Dice** | per subject, take your k highest-scored nodes where k = that subject's true positive count; report the mean |

Where you compare two models, report **mean ± standard deviation over at least 2
seeds**. A single-run difference is not evidence.

> *Optional:* sensitivity/specificity and Cohen's kappa at a threshold you justify.

---

## 5. Required analyses

### 5.1 Baseline before graphs 
Train **one non-graph model** that treats each node as an independent row (MLP,
logistic regression, gradient boosting — your choice). Tune it honestly on `val`.
This is not a throwaway; it is the bar your graph model must clear.

> *STRETCH:* a second baseline of a different family, to see whether
> non-linearity matters.

### 5.2 Graph model and ablations 
Train **one message-passing model** — GCN, GraphSAGE, GAT, or equivalent.
**Writing a simple propagation layer yourself is entirely acceptable** and needs
no special library; one propagation step is essentially "average your neighbours'
features, then classify." Do not compare multiple GNN architectures — we are not
asking for that.

Then answer, with evidence: **is the graph structure genuinely contributing, or is
the gain noise?** Run at minimum:

* **real adjacency** — your model
* **shuffled adjacency** — permute node order in the adjacency, preserving the
  degree distribution but destroying the feature–topology correspondence
* **identity adjacency** — no message passing at all

Report all three with seed variability, and state plainly what you conclude.

> **A correct conclusion of "the graph adds little," backed by good evidence,
> scores better than an unsupported claim that it helps.**

> *STRETCH:* vary propagation depth; randomise edge weights while keeping
> topology; sparsify by weight threshold.

### 5.3 Multimodal fusion and missing modalities
* Report **modality_A only**, **modality_B only**, and **both**. All three — A-only
  vs both tells you whether B adds anything *given* A, but only B-only tells you
  whether B is weak or merely redundant.
* Handle the 25% of subjects with **no modality_B**. Describe your strategy,
  implement it, and **measure it** — report test performance on
  modality_B-present and modality_B-absent subjects **separately**.
* Handle the scattered `NaN`s in modality_A. Say what you did and why.

Simple concatenation is fine. We are not asking for cross-attention or gated
fusion.

> *STRETCH:* compare two missing-modality strategies head to head.

### 5.4 The independent simulation score 
Treat `sim_score` as a **second, independent opinion** — not as a feature to dump
into your model on turn one. In this order:

1. **Evaluate it standalone.** AUROC/AUPRC on test subjects.
2. **Quantify concordance** with your model's output — correlation, rank
   agreement, or top-k set overlap. Report a number, not an impression.
3. **Examine the discordant cases.** Where do the two disagree most? Is there
   anything that characterises those subjects? **Does `sim_confidence` explain the
   disagreement?** Show the evidence.
4. **Only then build a fusion.** Compare at least: your model alone, the
   simulation alone, a naive average, and **one fusion smarter than a naive
   average**, justified by what you found in steps 1–3.

> *STRETCH:* a learned/stacked fusion fitted on `train` + `val`.

### 5.5 Outcome association 
`resected` marks nodes that received an intervention; `engel_1_seizure_free` is a
subject-level binary outcome. Test whether the overlap between the resected set
and your predicted abnormal set is associated with the outcome. Report an effect
size and a p-value, plus two sentences of interpretation.

### 5.6 Uncertainty *(STRETCH)*
Attach per-node uncertainty (ensemble spread, MC dropout, calibrated intervals)
and show one plot or table that uses it.

---

## 6. Deliverables

1. **`ASSIGNMENT.ipynb`** — completed, with every REQUIRED cell filled in and
   outputs saved, plus any `.py` modules you wrote.
2. **`predictions.csv`** — columns `subject_id`, `node_id`, `prob_abnormal`, for
   all **28 test subjects × 68 nodes = 1,904 rows**, from your single final model.
   Validate it with `check_data.verify_predictions()` before submitting.
3. **`README.md`** — setup and the command(s) that reproduce your main results
   table. Set random seeds.
4. **`REPORT.pdf` or `.md`** — **maximum 5 pages**, containing:
   * one **results table** covering every model in Section 5
   * your answer to **"does the graph help, and how do you know?"**
   * your **missing-modality** strategy and its measured effect
   * your **concordance/discordance** analysis and fusion rationale
   * short **"what I'd do with more time"** and **limitations** sections

---

## 7. Practical notes

* **CPU is enough.** ~9,500 rows, 140 small graphs. No GPU. If something takes
  more than a few minutes to train, you are overbuilding.
* **Environment:** `pip install -r requirements.txt`. Any additional library is
  allowed — just pin versions if it is awkward to install.
* **Suggested starting hyperparameters** (adjust if you wish, minimal tuning is
  fine): hidden dim 64, learning rate 1e-3, ~100 epochs, 2 propagation layers.
* Do **not** use `resected` or `engel_1_seizure_free` as inputs or targets.
* No external datasets.
* **AI coding assistants are permitted.** You must understand and be able to
  defend every line; we will ask about specific choices at interview.
* If something is ambiguous, make a reasonable assumption, **write it in the
  report**, and continue. Documented assumptions are never penalised.

### Suggested time budget (~10 h)



---

## 8. How we grade

| Weight | Criterion |
|---|---|
| 25% | **Experimental judgement** — honest comparisons, seed variability, no leakage, conclusions that match the evidence |
| 20% | **Graph reasoning** — the ablations in 5.2 and what you conclude |
| 20% | **Concordance & fusion** — 5.4 done in the right order, not a blind average |
| 15% | **Multimodal & missing data** — strategy *and* its measurement |
| 10% | **Code quality & reproducibility** |
| 10% | **Communication** — clarity and honesty about limitations |

Note what is *not* on that list: raw leaderboard score. **We care far more about
how you reached your conclusion than about the third decimal place.**

Good luck.
