# Stage H: hardening the v2 results

Date: 2026-11-17. Follows Stages A0, A4 and B1 in
[the v2 design](2026-11-14-federated-fraud-v2-design.md). Branch `stage-a0-core-pipeline`, local commits only.

## Intent

The headline result (A4) is that account-level DP at ε=8 keeps 98.8% of the best no-DP PR-AUC on IBM
HI-Small once the model is pretrained on public data. A reviewer would attack it on six points:

1. Private training adds at most 0.01 over the pretrained model alone (0.553 → 0.558–0.565).
2. The public data (LI-Small) and the private data (HI-Small) come from the same simulator.
3. One seed for A0 and one pretraining run, so most rows have no error bars.
4. Choosing the best round on HI-Small validation sits outside ε.
5. DP and SecAgg+ were never run together; the DP noise comes from a seeded generator.
6. Smaller issues: FedProx's μ is untuned, client 0 scores ~0.02 everywhere, false-alert volume is not
   reported, and the comparison with published HI-Small results is from memory.

Stage H fixes these before any new feature work. Order agreed with the user: weak points (this stage),
then strong points (smaller ε, graph features), then Kubernetes (B2). The goal is a publishable
pipeline whose claims survive review.

## Goals

- **G1 Out-of-distribution public data.** Pretrain on SAML-D (a different generator), fine-tune on
  HI-Small exactly as A4 did, and report whether the DP result holds. The local AMLSim sample gives a
  second, weaker check.
- **G2 When does private data matter.** Pretrain on 10% and 1% of LI-Small's train accounts and
  measure the gain from private training (with and without DP) at each size.
- **G3 Error bars.** Two more LI-Small pretraining seeds (1, 2), each followed by the fine-tune sweep.
  Seeds 1–2 for the A0 rows the headline is compared with: `centralized`, `gbdt`, `fedavg`.
- **G4 Model selection inside the privacy budget.** Every fine-tuning setting (learning rate, which head
  layers train, number of rounds) is chosen on public data only, then fixed before any HI-Small run.
  The HI-Small model is the one after the fixed number of rounds: no selection on HI-Small validation.
- **G5 DP + SecAgg+ together.** Run the A4 head-DP fine-tuning as a Flower app with SecAgg+.
- **G6 Secure DP noise.** DP noise comes from the operating system's cryptographic random source.
- **G7 Smaller fixes.** Tune FedProx μ; explain client 0; add alert-budget metrics; check published
  HI-Small results.

## Non-goals

- New model features (graph branch, cross-bank features): that is the strong-points stage.
- DP on backbone layers. Per-example gradients through the backbone are what made A0's DP fail, and they
  remove the embedding cache that makes head DP cheap. "More of the model" in this stage means the whole
  classifier head (1,089 parameters) instead of its last layer (33).
- Re-running A0 seeds for the DP, `local` and `fedprox` rows. The DP rows sit at chance in every seed;
  `fedprox` is replaced by the μ sweep.
- Changing client 0's composition. This stage explains it; a fix is a later decision.
- Kubernetes, encryption benchmarks, inference latency.

## Anti-goals (a failure even if everything runs)

- Tuning on HI-Small until the numbers look good. Settings are fixed on public data and written into the
  JSON before the HI-Small runs; HI-Small validation is used only for reporting.
- Dropping or softening a result that weakens the headline. If SAML-D pretraining breaks the DP result,
  the write-up says so and the README claim narrows to "same-generator public data".
- Mixing seeded and secure-noise numbers in one table without saying which is which.
- Committing raw data, `*.pt` files or anything that exposes the user's personal details.

## Constraints

- One MacBook, 8 CPU cores, 8 GB RAM: one training job at a time. Data on the external drive; logs, JSON
  and checkpoints of long runs on the internal disk (`~/capstone_runs/`), copied into the repo after.
  Detached runs with `setsid` + `caffeinate`.
- Commits stay local; never push. Long runs are never restarted from scratch; only missing parts rerun.
- Features stay fixed transforms (nothing fitted on data), so no statistics of one dataset leak into
  another.
- δ = 1e-6, account-level DP, RDP accounting, 10 clients, chronological 60/20/20 split — as in A0/A4.
- SAML-D: `/Volumes/Untitled/datasets_capstone/SAML-D.csv` (9,504,852 rows, 855,460 accounts, 0.104%
  laundering, 2022-10-07 to 2023-08-23). AMLSim sample: `/Volumes/Untitled/datasets_capstone/ibmamlsim/`
  (1,323,234 transactions, 10,000 accounts, one bank, one currency, one transaction type, 200 time steps).

## Approaches considered

**Running the stage.**
(a) One plan: code changes first, each test-driven, then the long runs queued back to back. **Picked.**
(b) Separate mini-stages with their own gates: more review overhead, and the runs could not be batched
overnight.

**Keeping model selection inside ε (G4).**
(a) **Cross-tuning on public data. Picked.** Each backbone's fine-tuning settings are tuned on the
*other* public dataset, which it has never seen, split into 10 pseudo-banks: the LI-Small backbone is
tuned on SAML-D, the SAML-D backbone on LI-Small. Nothing private is touched, so no ε is spent and the
tuning set really is unseen by the backbone.
(b) A private selection mechanism (noisy argmax over rounds): needs a sensitivity bound for PR-AUC under
account-level neighbours, which is hard to state honestly.
(c) Always take the last round: simple, but arbitrary, and in A4 8 of 9 runs peaked at round 1.

**Out-of-distribution public data (G1).** SAML-D (picked, user downloaded it) is closest to the IBM
schema; the AMLSim sample is a second check. Generating a larger AMLSim run was rejected: it needs Maven
and is still schema-poor.

**Secure noise (G6).** Draw uniform bits from `os.urandom`, turn them into Gaussians with Box–Muller in
float64, and use Opacus's secure-mode guard against floating-point attacks (the mean of several Gaussian
draws, scaled to the target σ). **Picked.** Opacus's own secure mode needs `torchcsprng`, which no longer
builds for current PyTorch. A seeded mode stays for tests and is recorded in each JSON.

## Decisions

### Data mapping (my calls, accepted by the user)

| IBM field | SAML-D | AMLSim sample |
|---|---|---|
| timestamp | `Date` + `Time` | `TIMESTAMP` step × 86,400 s from 2022-01-01 |
| bank (from/to) | sender / receiver bank location | one bank for all |
| account | `Sender_account` / `Receiver_account` | `SENDER_ACCOUNT_ID` / `RECEIVER_ACCOUNT_ID` |
| amount paid / received | `Amount` for both | `TX_AMOUNT` for both |
| currencies | matched by name to IBM's list (UK pounds → UK Pound, Euro, Swiss franc, Yen, US dollar, Mexican Peso, Indian rupee and Pakistani rupee → Rupee); others unknown (all-zero one-hot) | US Dollar |
| payment format | Credit card, Debit card → Credit Card; Cheque; ACH; Cross-border → Wire; Cash Deposit, Cash Withdrawal → Cash | ACH |
| label | `Is_laundering` | `IS_FRAUD` |

Both are built with the existing builder into the same indexed format, chronological 60/20/20, one client
for pretraining. For tuning, SAML-D is also built with 10 clients by hashing accounts (it has no banks),
and LI-Small with 10 clients by the existing bank grouping. Overlap with HI-Small: impossible by
construction (different generators and account ID spaces); the write-up states this instead of running
the row-hash check.

**Amendment (2026-11-17, approved before any HI-Small run).** The LI-Small backbone tuned on SAML-D was at
chance in all 96 grid runs (val PR-AUC 0.0010–0.0017 against a base rate of 0.00105, no DP included): its
features do not transfer to SAML-D, so the picks were noise. That grid is kept as a result. The LI-Small
backbone is instead tuned on IBM LI-Medium, which is public and which it never saw, cut to the rows of one
third of sending accounts (`experiments/subsample_senders.py`, crc32 of bank|account mod 3) so it is about
SAML-D's size and every kept account keeps its whole history. It is built with 10 clients by bank grouping.
LI-Medium shares the generator family with HI-Small, so the row and account overlap check is run and reported.

### Fine-tuning settings (G4)

- Tuning grid, at ε=1, 3, 8 and no DP, seed 0, on the tuning dataset's 10 pseudo-banks:
  learning rate {3e-4, 1e-3, 3e-3} × trained layers {last layer, whole classifier head} × total rounds
  R ∈ {1, 2, 5, 10}. *(My call.)* Each R is its own run, with σ calibrated for exactly R rounds, so the
  chosen setting is the one that was actually run, and the HI-Small run at that R spends exactly the
  target ε. Stopping a 10-round run early was rejected: it would spend less than the target ε with noise
  sized for a different run. Pick the (lr, layers, R) with the best tuning-set validation PR-AUC at the
  end of the run, separately for each ε and for no DP.
- Round 0 (the starting model) is not a choice: R ≥ 1 always. *(My call.)* `zero_shot` is reported
  separately, so a run whose private training does nothing still shows up as head_dp ≈ zero_shot.
- The chosen settings are written to `results/hardening/tuned_settings.json` and committed before any
  HI-Small fine-tuning run of this stage.
- HI-Small runs train for exactly the chosen R and report the final model. They also report what A4-style
  validation selection would have picked, for comparison only.
- No-DP rows follow the same rule *(my call)*: `head_nodp` uses the no-DP settings from the public grid,
  so Q4 compares DP and no-DP under one selection rule. `full_nodp` is too expensive to tune (about 2 h
  per run); it keeps A4's settings (lr 1e-3, 10 rounds) and A4's validation selection, and is labelled
  an optimistic no-DP ceiling. Being optimistic makes the DP ratio harder to meet, not easier.
- The 10% and 1% LI-Small backbones, pretraining seeds 1–2 and the AMLSim backbone reuse the settings
  tuned for the full LI-Small backbone (seed 0). Re-tuning each would multiply the grid; the cost if wrong is a slightly
  pessimistic number for those rows.

### Reduced public data (G2)

The 10% and 1% backbones pretrain on a uniform random sample (seed 0) of LI-Small's *train* accounts,
using the existing `max_train_accounts` sampling; LI-Small validation and test stay whole *(my call)*.
This measures "less public data" as it would happen. At 1% (about 6,750 accounts and a few dozen
laundering rows) the backbone may come out near chance; that is a result, not a failure, since it shows
where the private data has to carry the model.

### Verdicts fixed before the runs (Q1, Q4)

*(My call, for the user to check.)* A4's bar (half the best no-DP PR-AUC) is already cleared by
`zero_shot`, so it cannot fail. Stage H uses two pre-registered tests for each backbone:

- **DP keeps the utility:** head_dp_eps8 (mean over seeds) ≥ 95% of the best no-DP PR-AUC on the same
  backbone (head_nodp or full_nodp).
- **The result is useful:** head_dp_eps8 > A0 `centralized` (0.523, mean over its three seeds once G3
  is done), i.e. the private federated model beats pooling all raw data without public pretraining.

For Q1 (SAML-D), the README keeps "public pretraining makes account DP nearly free" only if both tests
pass on the SAML-D backbone. If only the first passes, the claim narrows to "DP costs little on top of a
pretrained model, but the model is only as good as its public data". If the first fails, it narrows to
same-generator public data. Q4 reports, per public-data size, head_dp_eps8 − zero_shot and
head_nodp − zero_shot with the seed spread; no threshold.

### Run matrix on HI-Small

| Backbone | zero_shot | head_nodp | full_nodp | head_dp ε=1,3,8 |
|---|---|---|---|---|
| LI-Small seed 0 (re-run with tuned settings) | seed 0 | seeds 0–2 | — (A4's value stands) | seeds 0–2 |
| LI-Small seeds 1, 2 | ✓ | seed 0 | — | seed 0 |
| LI-Small 10%, 1% of train accounts | ✓ | seeds 0–2 | seed 0 | seeds 0–2 |
| SAML-D | ✓ | seeds 0–2 | seed 0 | seeds 0–2 |
| AMLSim sample | ✓ | seed 0 | — | ε=8, seed 0 |

Plus A0 (no pretraining): `centralized`, `gbdt`, `fedavg` seeds 1–2; `fedprox` μ ∈ {0.001, 0.01} seed 0
(0.1 is A0's value). μ is chosen on HI-Small validation: this row has no DP, so that is allowed.
All A0 runs keep the 20,000-account cap.

### Flower + SecAgg+ with head DP (G5)

`run_flower.py` gains a head-DP mode: each client embeds its own data with the frozen backbone, trains
the head with account-level DP and the tuned settings, and SecAgg+ aggregates the head parameters only.
Like A4's DP runs, it uses every train account (no 20,000-account cap).
Gate: every round's aggregate within the quantization bound (as B1), and pooled test PR-AUC within 0.005
of the in-process run with the same backbone, ε=8, seed 0 (different per-client RNG streams, as in B1).

### Secure randomness (G6)

All DP randomness in this stage's new runs is secure: the Gaussian noise, the Poisson sampling of
accounts, and the choice of each sampled account's rows *(my call: the privacy-amplification argument
assumes the sampling is secret, and Opacus's secure mode covers sampling too)*. Non-DP randomness
(initialisation, dropout, shuffling in no-DP runs) stays seeded. Seeded DP randomness is opt-in
(`--seeded-dp`) and used by tests. Each JSON records `dp_randomness: "secure" | "seeded"`. Existing
A0/A4/B1 JSONs are not rewritten; tables mark them seeded.

### Metrics (G7)

`fraud_metrics` adds precision and recall when the top 1,000 and top 5,000 scored test transactions are
alerted (pooled). New runs report them; old JSONs are not back-filled.

### Client 0 (G7)

A written analysis, no model change: bank composition, laundering rate per split, how much of client 0
the largest bank is, and PR-AUC per bank inside client 0 for the Stage H head_dp_eps8 model (LI-Small
seed 0 backbone, seed 0). That run saves its per-transaction test scores for this purpose.

### Published results (G7)

Find the published HI-Small results (Multi-GNN, boosting with graph features), record their metric,
split and number with citations, and state how comparable ours are. If their metric is minority-class F1
on a split we can reproduce from our data, compute ours the same way.

## Deliverables

- Code: SAML-D and AMLSim loaders, account-hash client assignment, secure noise, alert-budget metrics,
  tuning script, `run_dp_finetune.py` fixed-rounds mode, `run_flower.py` head-DP mode, `pretrain_public.py`
  account-fraction option. All test-driven; suite green.
- Results: `results/hardening/*.json`, `results/hardening/HARDENING_GATE.md` answering:
  Q1 does the DP result hold with SAML-D pretraining; Q2 head_dp_eps8 under public-only tuning against
  A4's 0.558; Q3 spread across pretraining seeds; Q4 gain from private data at 100%/10%/1% public data;
  Q5 Flower + SecAgg+ head DP against in-process; Q6 A0 baselines with three seeds and tuned FedProx;
  plus client 0, alert budgets and the literature table.
- README and v2 spec updated, claims narrowed wherever a result says so.

## Compute budget (estimates, unloaded machine)

| Work | Hours |
|---|---|
| SAML-D build + pretraining (5 epochs) | ~2.5 |
| Cross-tuning grids (two backbones × 4 DP levels × 6 settings × R ∈ {1,2,5,10}) | ~8 |
| LI-Small seed 0 re-run with tuned settings | ~1.5 |
| SAML-D fine-tune sweep incl. full_nodp | ~4 |
| LI-Small 10% and 1% pretraining + sweeps | ~5 |
| LI-Small pretraining seeds 1–2 + sweeps | ~5 |
| AMLSim build, pretrain, short sweep | ~1 |
| Flower head-DP run | ~1 |
| A0 seeds 1–2 (centralized, gbdt, fedavg) | ~8.5 |
| FedProx μ sweep (2 runs) | ~4 |
| **Total** | **~40** |

## Left to the builder

- Module and function names, file layout inside `src/data/` and `experiments/`.
- How runs are queued (one shell chain per night is fine) and how JSONs are merged.
- The exact number of Gaussian draws in the floating-point guard, as long as it matches Opacus.
- Tie-breaking in the tuning grid (prefer fewer rounds, then the last-layer setting).
