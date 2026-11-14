# Federated Fraud Detection v2: Design and Roadmap

Date: 2026-11-14. Status: agreed in discussion, not yet built.

## Goal

A cross-bank fraud model that is trained without sharing raw data, has a privacy
guarantee small enough to defend (single-digit epsilon), and reports numbers that
are valid on honest splits. The target is a publishable result, which means every
claim has a measurement and a baseline next to it.

## Why v1 is not enough

| Problem in v1 | Evidence | Fix in v2 |
|---|---|---|
| SMOTE applied before the split | Row counts predict client 9's fraud count to within 1 (93,869 predicted, 93,868 in file) | No SMOTE; imbalance handled in training only |
| Train/test leak | `load_and_fix_client_data` pools all splits and re-splits at random | Chronological split, used as delivered |
| Sequences are not account histories | Pipeline slid a window over the whole time-sorted table (inferred, the script is not in the repo) | Per-account causal histories |
| Graph branch has no edges | `SimpleGraphEncoder` is an MLP | Stage A2 |
| Client-level DP with 10 clients | ε=1 destroys the model; real cumulative cost is ε≈3630 over 15 rounds | Account-level DP-SGD inside each bank |
| PaySim has no bank identifier | Clients are arbitrary chunks | IBM AML data, clients are groups of real banks |
| Flower path never ran; Paillier costs 31–48 s per client per round | `docs/ARCHITECTURE.md` | Track B |

## Decisions

1. **Primary dataset: IBM AML HI-Small.** Each transaction names the sending and
   receiving bank. A transaction is owned by its sending bank.
2. **Clients: 10 groups of whole banks**, balanced by transaction count with a
   deterministic greedy rule. The count of banks is not known until the file is
   profiled, and this rule works for any count.
3. **Split: chronological, 60/20/20, global time cut-offs.** Believed to match the
   split used by the published Multi-GNN baseline. Confirm in the paper before
   claiming comparability.
4. **Example = one transaction plus the sending account's previous 9 sent
   transactions.** Same tensor contract as v1: `(batch, 10, F)` and one label.
5. **Features use no statistics fitted on private data.** Log transforms with
   fixed divisors, cyclic time features, and one-hot vocabularies taken from the
   public schema. A fitted scaler would leak and would need its own privacy cost.
6. **Privacy unit: the account (customer), not the transaction.** One transaction
   appears in up to 10 examples through the history window, so per-transaction
   clipping would not bound its influence. Clipping per account does, and it is
   the stronger guarantee. All features in Stage A0 depend only on the sending
   account's own transactions, which is what makes this unit valid.
7. **DP mechanism: Poisson sampling of accounts, per-account gradient clipping,
   Gaussian noise, Rényi accounting per bank across all rounds.** Per-example
   gradients come from `torch.func` on the unmodified model. Opacus is used for
   accounting only: its hook-based engine fails on this model (bare `cls_token`
   parameter, tuple outputs, buffers in the contrastive head).
8. **Under DP the contrastive and auxiliary heads are frozen** and the loss is
   weighted BCE. NT-Xent couples examples in a batch, so it has no per-account
   sensitivity bound. Their loss weights are already 0 in `DEFAULT_CONFIG`.
9. **FedProx under DP is a decoupled proximal step** applied after the noisy
   update. It depends only on the current weights and the public global model,
   so it costs no privacy.
10. **Headline metrics: PR-AUC and recall at 1% false-positive rate.** Both are
    threshold-free. F1 is reported at a threshold tuned on validation data, with
    the caveat that the tuning is not covered by ε.
11. **Datasets are separate benchmarks, never merged.** Different features and
    different label meanings.
12. **Existing checkpoints, CLI flags and the web app keep working.** No
    state-dict key changes.

## Verified on this machine (M1, 8 GB RAM, Python 3.12, torch 2.8, flwr 1.25, opacus 1.5.4)

- Per-example gradients via `torch.func.vmap(grad(...))` work on the current
  45,292-parameter model: 512 examples in 0.25 s, matching autograd to 1.5e-8.
- Noise multipliers for account-level DP, δ=1e-6, 10 rounds × 2 local epochs,
  50,000 accounts per bank, 256 accounts per batch: σ=1.64 (ε=1), 0.88 (ε=3),
  0.63 (ε=8). These are ordinary DP-SGD noise levels.
- `xgboost` fails to import (missing `libomp`). Docker is not installed.
  `kubectl` and `kind` are installed. 26 GB free on the project drive.

## Not verified

- The IBM file's exact columns, category values, bank count and fraud count.
  Task 1 of the Stage A0 plan asserts them.
- Any accuracy number. Published centralized ceilings on HI-Small are about
  63–68% minority F1 (secondary sources). The estimate for this pipeline is
  55–65% without DP and lower with it.

## Architecture

```
Per bank group (later: one Kubernetes pod with its own volume)
  features.pt   per-transaction features, no fitted statistics
  history       index of each account's previous transactions (causal)
      │
  Sequence branch (existing TemporalTransformer)
  Graph slot      (A0: account's mean profile, as today; A2: bank-local GNN)
  Fusion → classifier (weighted BCE)
      │
  Training: account-level DP-SGD (sample accounts, clip per account, add noise)
      │   model update, already private
      ▼
  Server: FedProx-weighted average   (Track B: SecAgg+, mTLS, Kubernetes)
```

## Roadmap

Each stage has its own implementation plan, written when the stage before it has
produced results. Interfaces for later stages depend on measurements from
earlier ones, so they are not pinned here.

### Track A: results

| Stage | Deliverable | Exit gate |
|---|---|---|
| **A0 Core pipeline** | IBM data build, honest splits, account-level DP-SGD, baselines (each bank alone, centralized deep, centralized boosting), ε sweep | Federated beats the mean of banks-alone on PR-AUC; ε sweep table exists for client-level and account-level DP |
| **A1 Federated boosting** | Federated gradient-boosted trees on the same splits (needs `brew install libomp`) | Table row exists; decide whether the deep model stays the headline |
| **A2 Real graph branch** | Bank-local transaction graph, edge-aware GNN, capped neighbour sampling | Written statement of what the DP guarantee covers once neighbours' data enters an example; ablation with and without the branch |
| **A3 Personalisation** | Per-bank fine-tuning of the classifier head | Per-bank gain table, small banks reported separately |
| **A4 Public pretraining** | Pretrain on rebuilt PaySim, fine-tune on IBM with DP | Ablation at ε=3 and ε=8; paper text says pretraining data is public and outside ε |
| **A5 DP-FTRL** | Alternative DP optimizer that needs no sampling assumption | Same-ε comparison against DP-SGD |
| **A6 Cross-bank features** | A few per-account features computed across banks by secure sum or private set intersection | Gain measured; leakage of each feature stated |
| **A7 Secondary datasets** | IEEE-CIS and PaySim rebuilt from raw through the same builder; European CC as a tabular baseline | Same tables as A0 for each |

A2 and A7 were not in the earlier five-item list. A2 was part of the agreed
architecture and needs its own stage because it changes the privacy analysis.
A7 is the multi-dataset decision.

### Track B: deployment (changes no accuracy number)

| Stage | Deliverable | Exit gate |
|---|---|---|
| **B1 Flower + SecAgg+** | Training runs as Flower ServerApp/ClientApp with the SecAgg+ protocol | Same weights as the plain loop on a fixed seed, to float tolerance |
| **B2 Kubernetes** | Server pod and bank pods on `kind`, each bank with its own volume, network policies, mutual TLS | A round completes with one bank pod killed mid-round. Needs Docker or Colima; 8 GB RAM may cap the pod count |
| **B3 Encryption benchmark** | CKKS against the existing Paillier numbers, per round | Time and bytes per client per round, same model |
| **B4 Inference** | Local inference latency, p50 and p99, plain and int8 | Measured numbers replace the 200 ms claim |
| **B5 App and paper** | Backend and frontend read the new results; paper tables regenerated | No number in the app or paper lacks a JSON file behind it |

Track B can start after A0. B1 comes before B2.

## If A0's privacy cost is too high

If account-level DP at ε=8 keeps less than half the no-DP PR-AUC, try in order:
more accounts per batch; a higher sampling rate for accounts with a fraud label,
accounted at the highest rate; then pull A4 (pretraining) forward.

## Assumptions the paper must state

- Banks and the server follow the protocol but may try to learn from what they
  receive. Poisoned updates are out of scope.
- The guarantee covers each sending account's outgoing transactions at its own
  bank. It does not cover an account's role as a receiver.
- Hyperparameter and threshold tuning, and choosing the best round on validation data, are not counted in ε.
- Per-bank transaction and account counts (used for aggregation weights and the DP sampling rate) and the pooled training fraud rate (used to initialise the classifier bias) are treated as public.
- The Kubernetes deployment is a single-machine simulation.
