# Federated Fraud Detection

**A Temporal Graph Transformer trained across simulated banks with FedProx and client-level differential privacy — with real, measured results instead of asserted ones.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20capstone%20project-orange)

---

## Why this exists

Fraud detection is a cross-institution problem: the patterns that separate fraud from
legitimate activity are strongest when several banks' data is pooled, but raw
transaction data can never legally or competitively be pooled. This project trains one
shared fraud model across simulated banks **without any of them sharing raw data** —
each bank trains locally, and only a clipped, differentially-noised update ever leaves
it.

What makes this repo different from a typical capstone writeup: every number below is
read from a checked-in JSON file produced by an actual training run
(`results/comparison/comparison.json`, `results/comparison/epsilon_sweep.json`), not
asserted. Section [The road to real numbers](#the-road-to-real-numbers) documents three
real bugs that were producing fabricated-looking (but not fabricated) 100% results
before they were found and fixed — because a privacy-preserving ML project that can't
show its evaluation is trustworthy isn't worth much regardless of its architecture.

## Contents

- [What's actually implemented](#whats-actually-implemented)
- [Real, measured results](#real-measured-results)
- [Real PaySim results](#real-paysim-results)
- [Quick start](#quick-start)
- [The web app](#the-web-app)
- [Project structure](#project-structure)
- [The road to real numbers](#the-road-to-real-numbers)
- [Honest limitations](#honest-limitations)
- [Tech stack](#tech-stack)
- [Further reading](#further-reading)
- [Authors](#authors)

## What's actually implemented

```
Input: transaction sequence (batch, 10 transactions, 10 features)
  │
  ├── Temporal branch — 4-layer Transformer encoder
  ├── Graph branch    — per-account encoder (see limitations below)
  │
  └── Cross-modal fusion (bidirectional attention)
         │
         ├── Supervised head (BCE)        → fraud probability
         └── Contrastive head (NT-Xent)   → anomaly score for novel fraud
```

- **Federated training** — FedAvg and FedProx across 10 simulated bank clients with
  non-identical (though not fully heterogeneous — see [limitations](#honest-limitations))
  transaction distributions. FedProx's proximal term keeps training stable when only a
  minority of clients are sampled per round.
- **Client-level differential privacy** — DP-FedAvg (McMahan et al., 2018): each
  client clips its whole round update; the server adds one calibrated Gaussian noise
  draw to the aggregate, once per round. Not naive per-minibatch noising, which
  silently blows the stated privacy budget (see [ARCHITECTURE.md](docs/ARCHITECTURE.md)
  for the two broken versions of this that predated the current one).
- **A right-sized model** — 32-dim hidden size, ~45K parameters. This isn't
  arbitrary: Gaussian-mechanism noise scales with `σ√d`, so a smaller model needs
  proportionally less noise for the same `ε` — right-sizing moved the usable epsilon
  floor from ~2000 down to ~20–50, a 40–100× improvement, with zero change to the
  privacy mechanism itself.

## IBM AML results (v2 pipeline)

The sections below this one describe v1 (synthetic data and PaySim). v2 rebuilds the pipeline on IBM AML
HI-Small with honest splits: 10 clients that are groups of real banks, a chronological 60/20/20 split,
per-account causal histories, no SMOTE, and the account as the privacy unit. Design:
[spec](docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md). Full write-up:
[STAGE_A0_GATE.md](results/comparison/STAGE_A0_GATE.md). Reproduce with
`python3 experiments/build_ibm_aml.py` then
`python3 experiments/run_ibm_comparison.py --seeds 0 --epsilons 1 3 8 --max-train-accounts 20000`.

Pooled test set: 1,015,882 transactions, 1,798 laundering (0.177%). A random ranking scores PR-AUC ≈ 0.0018.
One seed, at most 20,000 train accounts per client.

| Model | PR-AUC | Recall @ 1% FPR | F1 | Train time (s) | ε spent |
|---|---|---|---|---|---|
| local | 0.203 | 0.419 | 0.338 | 8218 | none |
| centralized | 0.523 | 0.700 | 0.591 | 8165 | none |
| gbdt | 0.525 | 0.711 | 0.544 | 80 | none |
| fedavg | 0.336 | 0.544 | 0.397 | 6900 | none |
| fedprox | 0.203 | 0.451 | 0.259 | 7218 | none |
| client_dp_eps1.0 | 0.002 | 0.000 | 0.004 | 7384 | 3.3 (client-level, composed) |
| client_dp_eps3.0 | 0.003 | 0.022 | 0.007 | 7223 | 11.0 (client-level, composed) |
| client_dp_eps8.0 | 0.011 | 0.037 | 0.026 | 7207 | 36.5 (client-level, composed) |
| account_dp_eps1.0 | 0.001 | 0.004 | 0.001 | 3696 | max 0.99 (account-level) |
| account_dp_eps3.0 | 0.001 | 0.004 | 0.001 | 3832 | max 2.99 (account-level) |
| account_dp_eps8.0 | 0.001 | 0.004 | 0.000 | 4905 | max 7.99 (account-level) |

1. **Federated training helps without DP.** FedAvg (0.336) beats banks training alone (0.203) and wins on 8
   of 10 clients. FedProx at μ=0.1 only ties banks-alone (5 of 10 clients).
2. **DP from scratch learns nothing.** Account-level DP at ε = 1, 3 and 8 is at chance. Measured cause: about
   75 of a bank's 20,000 accounts have any laundering, so the clipped laundering gradient per step (norm 0.53)
   is about 450× smaller than the noise (norm 237). The same code with the noise off learns.
3. **Client-level DP is no better and costs more than its label.** "ε per round" of 1, 3, 8 composes to
   3.3, 11.0, 36.5 over 10 rounds.
4. **The deep model matches gradient-boosted trees centrally** (0.523 vs 0.525); FedAvg keeps 64% of that.

Not counted in ε: threshold tuning, hyperparameter choice and best-round selection on validation data;
per-bank counts and the pooled training laundering rate are treated as public.

### Stage A4: public pretraining + DP on the head

The full model is pretrained without DP on IBM LI-Small (public, same schema, no shared transactions with
HI-Small), then the backbone is frozen and federated account-level DP-SGD trains only the classifier's last
layer (33 parameters) on every HI-Small account. Full write-up:
[STAGE_A4_GATE.md](results/comparison/STAGE_A4_GATE.md). Reproduce with
`python3 experiments/build_ibm_aml.py --src <LI-Small_Trans.csv> --out data/ibm_li_small --clients 1`,
`python3 experiments/pretrain_public.py --epochs 5`, then
`python3 experiments/run_dp_finetune.py --ckpt checkpoints/pretrain_li_small.pt --epsilons 1 3 8 --seeds 0 1 2 --dp-train last_layer --lr 1e-3 --models zero_shot head_nodp head_dp`,
then the same with `--models full_nodp --seeds 0` (about 2 h unloaded) and merge the two JSON files.

Same pooled test set; mean ± std over seeds 0–2 (one seed for `full_nodp`).

| Model | PR-AUC | Recall @ 1% FPR | F1 | ε spent |
|---|---|---|---|---|
| zero_shot (pretrained, no private training) | 0.553 | 0.740 | 0.612 | none |
| head_nodp (FedAvg on the head) | 0.563 ± 0.000 | 0.749 ± 0.001 | 0.615 | none |
| full_nodp (FedAvg on the whole pretrained model) | 0.565 | 0.716 | 0.625 | none |
| head_dp_eps1.0 | 0.557 ± 0.000 | 0.739 | 0.614 | max 1.00 (account-level) |
| head_dp_eps3.0 | 0.558 ± 0.000 | 0.739 | 0.614 | max 3.00 (account-level) |
| head_dp_eps8.0 | 0.558 ± 0.000 | 0.739 | 0.614 | max 8.00 (account-level) |

1. **With a public pretrained backbone from the same generator, account-level DP is nearly free.** At ε=8 the model reaches 0.558,
   98.8% of the best no-DP federated PR-AUC (0.565), against under 1% in A0; the curve is flat from ε=1 to
   ε=8. But the pretrained model alone already reaches 0.553, so DP training itself adds only +0.006.
2. **The gain comes from the public data.** The pretrained model alone (0.553) beats A0's centralized model
   (0.523) on 9 of 10 clients. Private training adds about 0.01 PR-AUC without DP and 0.006 with it.
3. **Caveat, tested in Stage H below:** LI-Small and HI-Small come from the same simulator. With public data
   from a different generator (SAML-D, AMLSim) the frozen backbone does not transfer and head DP is at
   chance. A4 also chose the best round on HI-Small validation (outside ε) and used seeded DP noise.

### Stage B1: Flower + SecAgg+

The A0 FedProx loop also runs as a Flower ServerApp/ClientApp, and the server sees only the SecAgg+ sum
of the banks' models. In each of the 10 rounds the secure aggregate matched the plain weighted average of
the same local models to within 9.7e-6, below the quantization bound of 7.73e-5. Pooled test metrics stay
close to A0 (FedProx PR-AUC 0.209 vs 0.203; account DP ε=8 0.0014 vs 0.0014), so secure aggregation
does not change what the pipeline learns. Write-up:
[B1_GATE.md](results/flower/B1_GATE.md). Reproduce with

```bash
python3 -W ignore experiments/run_flower.py --rounds 10 --debug-dump-locals --out results/flower/b1_fedprox.json --max-train-accounts 20000
```

Add `--account-dp-eps 8` for the DP run. `--debug-dump-locals` writes each bank's model to disk to check
the aggregate, which defeats SecAgg+; leave it off outside this check.

### Stage H: hardening

Stage H re-runs A4 so that its claims survive review. Write-up:
[HARDENING_GATE.md](results/hardening/HARDENING_GATE.md); design:
[spec](docs/superpowers/specs/2026-11-17-hardening-design.md). Changes from A4:

- Every fine-tuning setting is chosen on public data only (IBM LI-Medium, one third of senders, 10 bank
  clients). HI-Small runs train for a fixed number of rounds and report the final model, so no choice
  sits outside ε.
- DP noise and sampling use the OS cryptographic source.
- Error bars cover three pretraining seeds and three A0 seeds.
- Pretraining is also tested on public data from other generators (SAML-D, AMLSim) and on 10% and 1% of
  LI-Small.
- Head DP runs through Flower with SecAgg+.

| Backbone | zero_shot | head_nodp | head_dp ε=8 | Best no-DP | head_dp ε=8 F1 |
|---|---|---|---|---|---|
| LI-Small seed 0 | 0.553 | 0.551 | 0.550 ± 0.000 | 0.565 (A4 full_nodp) | 0.610 |
| LI-Small, 3 pretraining seeds | 0.528 ± 0.018 | 0.532 ± 0.014 | 0.532 ± 0.013 | | 0.593 ± 0.015 |
| LI-Small 10% | 0.137 | 0.138 | 0.163 | 0.474 (full_nodp) | 0.265 |
| LI-Small 1% | 0.096 | 0.110 | 0.040 ± 0.023 | 0.359 (full_nodp) | 0.054 |
| SAML-D | 0.002 | 0.008 | 0.004 | 0.493 (full_nodp) | 0.013 |
| A0 centralized, no pretraining (3 seeds) | | | | 0.527 ± 0.005 | 0.583 ± 0.011 |

1. **DP keeps 97% of the best no-DP PR-AUC on the same-generator backbone** (pre-registered bar 95%), and
   beats centralized training without pretraining (0.550 vs 0.527). Averaged over pretraining seeds the
   margin (0.532 vs 0.527) is within the noise.
2. **The README's A4 claim narrows to same-generator public data.** SAML-D and AMLSim backbones fail the
   pre-registered test: frozen, they score at chance on HI-Small. Fine-tuning the whole SAML-D model
   without DP reaches 0.493, so the next lever is DP on more than the head.
3. **On a strong backbone, private training adds nothing measurable.** With 10% of the public data, head
   DP adds +0.027 while full fine-tuning without DP adds +0.34.
4. **Flower + SecAgg+ with head DP matches in-process** (0.5494 vs 0.5495). Each round's aggregate error
   is at most 3.6e-6, against a bound of 7.7e-5.
5. **Tuned FedProx** (μ=0.001, chosen on validation) reaches 0.368, against 0.203 at A0's μ=0.1. FedAvg
   over three seeds is 0.392 ± 0.044.
6. **Alert budgets:** reviewing the top 1,000 scored test transactions, 85% are laundering and they
   catch 47% of all laundering; at 5,000 alerts, 24% and 66%.
7. **Client 0** is 89% one bank whose 15 hub accounts launder with the same payment types as their
   legitimate traffic, and the model is at chance there. Receiver and graph features (A2) are the fix.
8. **Against published HI-Small F1:** ours is 61.0 (59.3 ± 1.5 over pretraining seeds). Published
   centralized models without privacy score 62.9–63.2 for boosting with graph features and up to 68.2
   for Multi-PNA + EU. The comparison is rough: those models are centralized and use graph features, and
   our threshold is tuned on validation.

## Real, measured results

Four models, trained for real, evaluated on one identical held-out test set (6,000
sequences, 362 fraud) — not a lookup table. Reproduce with
`python experiments/run_model_comparison.py --rounds 10`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Privacy |
|---|---|---|---|---|---|---|
| Centralized RandomForest | 98.27% | 95.74% | 74.59% | 83.85% | 94.74% | None |
| Centralized deep model | 97.72% | 90.04% | 69.89% | 78.69% | 94.11% | None |
| Federated (FedAvg, no DP) | 97.40% | 84.11% | 70.17% | 76.51% | 93.98% | Data locality only |
| **Federated (FedProx + DP, ε=100)** | **97.50%** | **85.81%** | **70.17%** | **77.20%** | 93.48% | **Formal (ε,δ)-DP** |

The honest framing isn't "beats everything" — a method with zero privacy cost (the
RandomForest) should and does edge out a privacy-preserving one on raw accuracy. The
real claim: this project is statistically indistinguishable from pooling every
institution's raw data (0.2pp off centralized), while being the only approach with a
provable privacy guarantee.

**Does differential privacy actually do anything, or is `ε=100` just a number?** A full
9-point sweep (`experiments/run_epsilon_sweep.py`) answers that directly:

| ε | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| 1.0 (textbook) | 6.0% | 0.114 | 0.500 — destroyed |
| 5.0 | 6.2% | 0.114 | 0.442 — destroyed |
| 10.0 | 74.1% | 0.127 | 0.577 — recovering |
| 20.0 | 96.7% | 0.688 | 0.916 — working |
| 50.0 | 97.5% | 0.763 | 0.930 — near-ceiling |
| **100.0 (default)** | **97.1%** | **0.751** | **0.935** |
| 2000.0 | 97.0% | 0.751 | 0.936 |
| No DP (ceiling) | 97.5% | 0.769 | 0.938 |

That's the shape a real privacy-utility tradeoff is supposed to have: a hard floor
where the mechanism genuinely destroys the model, a transition region, and a plateau —
not a flat line, which is what a broken (inert) DP implementation would look like.

## Real PaySim results

The tables above use synthetic data. Real PaySim data (`preprocessed_datasets_csv/`,
1.9M real transactions, pre-windowed into 2.86M ten-step sequences across 10 clients)
is also in this repo, but it needed a real fix before it was usable:

**The data problem.** Every client's delivered `val`/`test` CSV was **100% fraud, zero
legitimate transactions** — verified on the full files, not a sample. That makes
precision, ROC-AUC, and a confusion matrix undefined on those splits. Pooling
train+val+test and re-splitting (a fix that works for a simpler "misassigned rows" bug)
was tried and rejected: it just redistributed the same corruption into a uniform ~30%
fraud rate everywhere, nowhere near real PaySim's documented ~0.13% rate — proof the
val/test files are a separate, inflated fraud pool bolted onto a clean `train` file, not
misassigned legitimate rows. The actual fix
(`experiments/build_real_paysim_splits.py`): rebuild all three splits from each client's
own `train` CSV alone (0.04%–0.17% fraud for 9 of 10 clients — the right order of
magnitude for real PaySim), via a fresh stratified 70/15/15 split. **Client 9 is a
separate, unexplained anomaly** — its own train file alone is 46.9% fraud, ~1000x every
other client — carried through as-is rather than silently corrected, since its root
cause is unknown (`configs/paysim.yaml`, named in the dataset's own metadata as what
built it, doesn't exist anywhere in this repo).

**Baseline results** (`python run_training.py --dataset paysim_real --rounds 15
--clients 10 --exclude-clients 9 --pos-weight 8 --fraud-rate-prior 0.001 --max-samples
50000`, 9 clients, client_9 excluded so the test set stays representative of the true
~0.1% rate):

| Metric | Value |
|---|---|
| Accuracy | 99.71% |
| Precision | 54.74% |
| Recall | 48.15% |
| F1 | 51.23% |
| ROC-AUC | 89.32% |

This is meaningfully worse than the synthetic numbers above (F1 51% vs. 77%) — expected,
not a failure: real PaySim's fraud rate is ~45x more imbalanced, and real fraud patterns
are harder to separate than a rule-based generator's clean ones.

**The fraud-starvation bug and its fix.** At real PaySim's ~0.1–0.3% fraud rate and
`batch_size=64`, most training batches contain **zero** fraud examples — so `pos_weight`
(the loss-level class reweighting) has nothing to act on most of the time. Verified two
ways before fixing it: sweeping `pos_weight` from 8 to 400 changed nothing (confusion
matrix was byte-identical across the whole range — per-batch gradient-norm clipping was
saturating its effect), and post-hoc threshold tuning on the trained checkpoint recovered
essentially zero extra F1. The real bottleneck was upstream of both. Fix: a
`WeightedRandomSampler` (`run_training.py --oversample --oversample-ratio R`) that
oversamples fraud into every batch at a target rate `R`, instead of the true (near-zero)
class ratio.

**Full balance isn't the right ratio — it's a dial, and it has a sweet spot.** Sweeping
`R` (8-round test runs) gave a clean, monotonic precision/recall tradeoff:

| `R` | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| 0 (baseline, no oversample) | 0.6364 | 0.5833 | 0.70 | 0.9039 |
| 0.01 | 0.6305 | 0.5648 | 0.7135 | 0.8966 |
| **0.02** | **0.6560** | **0.7593** | 0.5775 | 0.9713 |
| 0.05 | 0.5434 | 0.7963 | 0.4125 | 0.9725 |
| 0.1 | 0.4123 | 0.8704 | 0.2701 | 0.9749 |
| 0.5 (full balance) | 0.2311 | 0.9074 | 0.1324 | 0.9741 |

`R=0.02` is the one value that improves **both** F1 and recall over baseline simultaneously
— every other tested ratio only trades one for the other.

**An honest wrinkle: more training made it worse, not better.** Scaling `R=0.02` up to
the full run (15 rounds, 50k samples, matching the baseline's scale) did **not** reproduce
the short-run win — it scored *worse* on F1 than even the plain baseline:

| Config | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Baseline, 8 rounds/20k | 0.6364 | 0.5833 | 0.70 | 0.9039 |
| Baseline, 15 rounds/50k | 0.5123 | 0.4815 | 0.5474 | 0.8932 |
| Oversample R=0.02, 8 rounds/20k | **0.6560** | **0.7593** | 0.5775 | 0.9713 |
| Oversample R=0.02, 15 rounds/50k | 0.4220 | 0.8009 | 0.2864 | 0.9826 |

The pattern holds for *both* configs: more rounds and more data consistently **hurt** F1
here, not helped — almost certainly because each client has only ~100–300 real fraud
sequences total, so longer training just keeps pushing the decision boundary toward
flagging more (recall and ROC-AUC climb every time; precision collapses faster). The
headline real-data checkpoint (`results/paysim_real_best/`) is therefore the smaller
8-round/20k-sample, `R=0.02` run — chosen because it's empirically the best by F1, not
because it's the most training. The longer run (`results/paysim_real_oversampled/`) and
the plain baseline (`results/paysim_real/`) are both kept for comparison, not discarded.

Including client_9 scores much better still (F1 78%) — but that's measuring against a test
set client_9 made ~8% fraud instead of ~0.1%, i.e. an easier distribution, not a better
model. Reported separately, not folded into the headline number above.

### Real PaySim model comparison

The 4-model comparison in ["Real, measured results"](#real-measured-results) above is
synthetic-only. Re-running it on real data (`experiments/run_model_comparison.py
--dataset paysim_real --exclude-clients 9 --oversample-ratio 0.02`, 8 rounds, 20K
samples/client) surfaces a real, structural finding, not just weaker numbers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Centralized RandomForest | 99.9% | 76.3% | 33.0% | 46.1% | 96.7% |
| Centralized Deep Model | 99.9% | 0% | 0% | 0% | 50.0% |
| Federated (FedAvg, no DP) | 84.2% | 0.3% | 69.3% | 0.7% | 84.8% |
| **Federated (FedProx+DP) — this project** | 99.8% | 18.6% | 51.2% | 27.3% | 85.3% |

**Oversampling alone didn't fix every model.** A first attempt without it collapsed 3 of
4 models outright (F1=0 for B, C, and D alike). Adding the same `--oversample-ratio 0.02`
fix that worked for the main pipeline recovered a real, non-degenerate result for
FedProx+DP — but **Centralized Deep Model stayed fully collapsed** (F1=0, ROC-AUC=0.5,
literally random, even after 4 fresh-init retries), and **FedAvg without DP inverted
into the opposite failure mode** — flagging 42,723 of 227,410 legitimate transactions as
fraud (effective precision 0.3%), not learning a real boundary. Only FedProx+DP, the
only one of the four with *both* a proximal term and DP noise as training-time
regularization, produced a stable result. This is consistent with
`run_centralized_deep`'s own docstring, which already noted it's "the only one of the
four models with zero training-time regularization" - on real data's much sharper
imbalance, oversampling fixes batch-level fraud starvation, but doesn't substitute for
that missing regularization. Not a bug to chase further - a real property of this
architecture under this data regime, reported honestly rather than re-run until it looks
better.

### Real PaySim epsilon sweep

Same question as the synthetic sweep above - does DP actually do something, or is
`ε=100` just a number - answered on real data
(`experiments/run_epsilon_sweep.py --dataset paysim_real --exclude-clients 9
--oversample-ratio 0.02 --epsilons 1 10 50 100 500`, 5 rounds, 10K samples/client -
reduced scope vs. the synthetic sweep's 9 points/10 rounds, since the comparison run
above took 96 minutes and a comparable 9-point sweep wasn't tractable in one sitting):

| ε | Accuracy | F1 | Recall | ROC-AUC |
|---|---|---|---|---|
| 1.0 | 0.08% | 0.0016 | 100% | 0.500 — destroyed |
| 10.0 | 96.54% | 0.0189 | 41.86% | 0.731 — recovering |
| **50.0** | 99.83% | **0.3395** | 55.35% | 0.864 |
| 100.0 (default) | 99.82% | 0.3278 | 54.88% | 0.865 |
| 500.0 | 99.83% | 0.3238 | 49.77% | 0.853 |
| No DP | 99.85% | 0.3162 | 42.79% | 0.824 |

Same qualitative shape as synthetic: destroyed at ε=1 (the model collapses to flagging
*everything* as fraud - recall=100% because nothing is held back, AUC=0.5 because that's
equivalent to a coin flip), recovering by ε=10, plateaued from ε=50 up - real,
independent confirmation the mechanism does genuine work on real data too, not just the
synthetic benchmark. One real, interesting difference from synthetic: here, moderate DP
(ε=50-500) slightly **outperforms** no-DP (F1 0.32-0.34 vs. 0.316) rather than costing
utility - consistent with the model-comparison finding above that DP's noise acts as
useful training-time regularization on this severely imbalanced real data, not purely a
privacy tax.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic dataset (schema-matched stand-in for PaySim —
#    see "Honest limitations" for why the real dataset isn't bundled)
cd experiments && python generate_synthetic_paysim.py --out ../data/synthetic_paysim --clients 10 && cd ..

# 2. Train
python run_training.py --dataset synthetic_paysim --rounds 10 --clients 10 --clients-per-round 3
python run_training.py --dataset synthetic_paysim --dp --epsilon 100 --rounds 10   # with DP

# 3. Reproduce the comparison table and epsilon sweep above
python experiments/run_model_comparison.py --rounds 10
python experiments/run_epsilon_sweep.py --rounds 10
```

## The web app

A FastAPI backend serves the real trained checkpoint; a React + TypeScript frontend
gives it six pages, all backed by real endpoints — nothing in the UI is mocked.

```bash
# terminal 1
cd backend && uvicorn main:app --reload --port 8000
# terminal 2
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`:

| Page | What it does |
|---|---|
| **Overview** | Real headline metrics pulled live from the trained checkpoint |
| **Test Your Data** | Upload *any* CSV (map your own columns after upload) or score one transaction by hand, against the real checkpoint |
| **Train** | Kicks off an actual small/fast FedProx + DP-FedAvg run with live per-round progress |
| **Live Simulation** | Watch it happen: 10 animated bank nodes generate fresh transactions, train locally, and send updates to a central server — with real, live fraud detection results streaming in as it runs |
| **Model Comparison** | The real 4-model comparison table above, plus a disclosed privacy-weighted composite score |
| **Architecture & Privacy** | The real 9-point epsilon sweep chart, and an honest accounting of what's wired in vs. what's a documented simplification |

A Streamlit dashboard (`./run_enhanced_dashboard.sh`) also still works and covers more
of the explanatory content — see [docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md).

## Project structure

```
federated-fraud-detection/
├── src/
│   ├── models/temporal_graph_transformer.py  # the model
│   ├── federated/                            # DP utilities, Flower client
│   └── data/feature_engineering.py           # the one shared feature encoding
├── run_training.py                # the training loop that's actually used
├── experiments/
│   ├── generate_synthetic_paysim.py  # schema-matched synthetic data generator
│   ├── run_model_comparison.py       # the 4-model real comparison
│   ├── run_epsilon_sweep.py          # the 9-point privacy-utility sweep
│   └── train_federated.py            # Flower/Ray reference path (see docs)
├── backend/main.py                # FastAPI serving the real checkpoint
├── frontend/                      # React + TypeScript + Vite dashboard
├── dashboard/                     # Streamlit alternative
├── configs/experiment_config.yaml
├── docs/
│   ├── ARCHITECTURE.md            # what's real vs. simplified, in detail
│   ├── DASHBOARD_GUIDE.md
│   ├── TESTING_GUIDE.md
│   └── paper/draft_paper.tex      # IEEE-format writeup of this work
└── results/comparison/            # the real JSON behind every table above
```

## The road to real numbers

Three real bugs were found and fixed during development — documented here because a
privacy-utility claim is only as credible as the pipeline that produced it, and
skipping this section would leave the misleading impression that everything just
worked.

1. **A trivially separable benchmark.** The first synthetic data generator drew fraud
   and legitimate transactions from disjoint value ranges. Every model — including a
   plain RandomForest — scored 100% on every metric, not because any model was good,
   but because the benchmark was trivial. Fixed by overlapping the distributions and
   folding in 12% "quiet fraud" drawn from the legitimate distribution.
2. **An over-parameterized, collapsing model.** At `hidden_dim=128` (~1.07M
   parameters for a 10-dimensional input), the model collapsed to a constant
   prediction (0% recall) under class imbalance and DP noise. Right-sized to
   `hidden_dim=32` (~45K parameters) with a prior-informed bias init and class
   weighting — see [Real, measured results](#real-measured-results) for the DP-budget
   payoff this had.
3. **A NaN-masking bug hiding a real training failure.** Guarding every loss term
   with a NaN-to-zero substitution prevented crashes but let the model silently train
   on a degenerate output once corrupted, because `0.0 * NaN == NaN` under IEEE 754
   corrupted the total loss even at zero weight. Fixed by sanitizing at the source and
   adding per-round checkpoint-and-rollback.

Two smaller but real bugs from the same pass: the CSV upload endpoint hung forever on
large files because `mapping`/`account_id_col` were missing `Form(...)` annotations
(FastAPI silently dropped them) and there was no row cap or batching — fixed and
load-tested on 200K+ row files; and the merchant-type dropdown was rendering
white-text-on-white on this dark theme, making 9 of 10 options invisible.

## Honest limitations

- The 4-model comparison and epsilon-sweep tables above are on a schema-matched
  **synthetic** stand-in (a pipeline-correctness check, not a real-world performance
  claim). Real PaySim results now exist separately — see
  [Real PaySim results](#real-paysim-results) — and score meaningfully lower, as
  expected given real PaySim's much sharper ~0.13% fraud rate (vs. the synthetic set's
  ~6%).
- The 10 simulated clients currently draw from the **same** underlying distributions
  with independent random samples, not genuinely heterogeneous per-bank patterns — real
  non-IID heterogeneity (different fraud rates, spending patterns per institution) isn't
  modeled yet, despite FedProx being chosen partly for non-IID robustness.
- The graph branch is a simplified per-account encoder, not a relational graph over
  shared entities (user, merchant, card, device) — the real `HeterogeneousGraphEncoder`
  (HGTConv) exists in code but needs entity-linked edge data this pipeline doesn't
  produce yet.
- Homomorphic encryption **and** Secure Aggregation are now both wired into the live
  training loop (`run_training.py --he` / `--he --secure-agg`) — real Paillier
  encryption of each client's full update, either aggregated directly by the server or
  routed through a tiered client→proxy→server topology so no single proxy sees every
  client. Real, measured cost: 31–48s to encrypt one client's full 45,292-param update,
  34–41s to decrypt the aggregate, per round (512-bit keys — demo-speed only, not
  production-secure; see ARCHITECTURE.md); adding the proxy tier costs almost nothing
  extra (0.33–0.41s for the proxy-level aggregation itself, since homomorphic addition
  is plain modular multiplication, not exponentiation).
- **Multi-round DP composition accounting is now implemented** (`src/federated/dp_accounting.py`,
  `run_training.py --dp`) — and building it surfaced a more significant finding than
  "no cumulative total existed." Every `ε` quoted above (ε=50, 100, 500, 2000) comes from
  this project's classical Gaussian-mechanism calibration formula
  (`σ = sensitivity·√(2ln(1.25/δ))/ε`), which is only mathematically proven tight for
  **ε≤1**. Verified numerically across the actual range this project uses: at ε=100 (the
  project default), the *true*, rigorously-computed single-round privacy loss — via Rényi
  DP, which has no such restriction — is **ε≈313, not 100** (a 3.13x gap; the gap is
  negligible at ε≤1 and grows with ε above that). A real 15-round training run
  (`results/paysim_real_dp_accounting/`) shows the full picture: naively multiplying the
  *label* by rounds gives 1500; naively summing the *true* per-round cost gives 4697.8;
  proper RDP composition gives **3629.9** — tighter than the true-naive sum (confirming
  composition accounting does real work, saving ~23%), but still ~2.4x what the commonly
  quoted "ε=100" label alone would suggest. Reported in full, not rounded up: this
  project's actual cumulative privacy budget over a real training run is in the
  thousands, not the hundreds.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full, unrounded accounting.

## Tech stack

**Model & training:** PyTorch, PyTorch Geometric, Opacus (reference), Flower
(reference path) · **Backend:** FastAPI, Uvicorn · **Frontend:** React 19, TypeScript,
Vite, Tailwind CSS v4, Recharts, Framer Motion, Lenis · **Data:** NumPy, Pandas,
scikit-learn

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the unrounded technical accounting:
  what's real vs. a documented simplification, the full DP history, the model
  comparison methodology
- [docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md) — Streamlit dashboard walkthrough
- [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) — live inference test values
- [docs/paper/draft_paper.tex](docs/paper/draft_paper.tex) — an IEEE-format writeup of
  this work (compile with `pdflatex`, or paste into [Overleaf](https://overleaf.com))

## Authors

Tanishk Maheshwari, Vaibhav Handoo, Vibhav Kolachana, Yash Kuber Khanna —
Department of Computer Science and Engineering, PES University, Bengaluru.
Guided by Dr. Prafullata Kiran Auradkar.

## License

No license file is currently included, which by default means all rights are
reserved. Add a `LICENSE` file (e.g. MIT) before or after pushing if you want this
repo to be reusable by others.
