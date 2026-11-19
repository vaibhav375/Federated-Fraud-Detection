# Stage H gate: hardened results

Date: 2026-11-19. Design: [hardening spec](../../docs/superpowers/specs/2026-11-17-hardening-design.md).
Every number below comes from a JSON file in this directory (or `results/comparison/` for A0/A4), except
the client 0 payment-type and account counts, which were computed from the raw HI-Small CSV during the
analysis and are not stored in `client0.json`.

All test metrics are on the pooled IBM HI-Small test set: 1,015,882 transactions, 1,798 laundering
(base rate 0.177%, so a random ranking scores PR-AUC ≈ 0.0018). F1 uses the threshold that maximises F1
on pooled validation scores, applied to test.

## What changed from A4

- **Settings chosen on public data only.** Learning rate, which head layers train and the number of
  rounds R come from a grid run on a public tuning set (`tuned_settings.json`, committed before any
  HI-Small run). HI-Small runs train for exactly R rounds and report the final model
  (`selection: "fixed_rounds"`). What A4-style validation selection would have picked is stored as
  `val_selected`, for comparison only.
- **Secure DP randomness.** Gaussian noise, Poisson account sampling and per-account row choice use the
  OS cryptographic source (`dp_randomness: "secure"` in every Stage H JSON). A0, A4 and B1 numbers were
  produced with the seeded generator and are marked "seeded" wherever they appear below.
- **Head DP trains the whole classifier head** (1,089 parameters) when the public grid picks it, which
  it did at every ε.

**Amendment, approved before any HI-Small run.** The first tuning set for the LI-Small backbone was
SAML-D. On it every setting scored at chance (validation PR-AUC 0.0010–0.0017, base rate 0.00105), so
those picks were noise. That grid is kept as a no-transfer result (`tune_li_small_on_saml_d.json`). The
backbone was retuned on IBM LI-Medium, cut to the rows of one third of sending accounts
(`experiments/subsample_senders.py`) and split into 10 bank clients. LI-Medium is public and unseen by
the backbone, but it shares HI-Small's generator, so the overlap was checked
(`overlap_li_medium.json`): 1 shared account, 0 shared transactions.

## Pre-registered verdicts (fixed in the spec before the runs)

- **DP keeps the utility:** head_dp_eps8 mean ≥ 95% of the best no-DP PR-AUC on the same backbone
  (head_nodp or full_nodp).
- **The result is useful:** head_dp_eps8 > A0 `centralized` three-seed mean, **0.5270**.

| Backbone (public pretraining) | head_dp_eps8 | Best no-DP | Ratio | DP keeps utility | > 0.5270 | Useful |
|---|---|---|---|---|---|---|
| LI-Small, seed 0 (3 fine-tuning seeds) | 0.5497 ± 0.0003 | 0.5652 (A4 full_nodp, seeded, val-selected) | 97.3% | **pass** | 0.5497 | **pass** |
| LI-Small, seed 1 | 0.5269 | 0.5217 (head_nodp) | 101.0% | pass | 0.52686 vs 0.52698 | fail (tie, −0.0001) |
| LI-Small, seed 2 | 0.5191 | 0.5226 (head_nodp) | 99.3% | pass | 0.5191 | fail |
| LI-Small, 10% of train accounts | 0.1633 ± 0.0015 | 0.4742 (full_nodp) | 34.4% | fail | | fail |
| LI-Small, 1% of train accounts | 0.0403 ± 0.0229 | 0.3585 (full_nodp) | 11.2% | fail | | fail |
| SAML-D (different generator) | 0.0041 ± 0.0000 | 0.4934 (full_nodp) | 0.8% | **fail** | | **fail** |
| AMLSim sample (different generator) | 0.0026 | 0.0028 (head_nodp) | 92% | fail | | fail |

Seeds 1–2 have no full_nodp run (the spec does not call for one), so their ratio is against head_nodp,
an easier bar than seed 0's.

The verdicts hold for the pre-registered backbone. Across three pretraining seeds, though, the "useful"
margin over centralized training (0.532 ± 0.013 vs 0.527 ± 0.005) is inside the seed spread.

## Q1. Does the DP result hold with SAML-D pretraining?

**No.** A model pretrained on SAML-D scores at chance on HI-Small zero-shot (0.0022). Training the head
on HI-Small, with or without DP, does not rescue it: head_nodp reaches 0.0080 and head_dp at ε=1/3/8
reaches 0.0038/0.0040/0.0041. A frozen SAML-D backbone does not produce features that separate
HI-Small laundering. Fine-tuning the whole SAML-D-pretrained model without DP reaches 0.4934, close to
A0 centralized (0.527), so the pretrained weights are a reasonable starting point. The frozen
representation is what fails to transfer.

The AMLSim sample backbone fails the same way (zero-shot 0.0017, head_nodp 0.0028, head_dp_eps8
0.0026). Tuning in the other direction also failed: SAML-D-backbone settings tuned on LI-Small topped out
at validation PR-AUC 0.0060 (`tune_saml_d_on_li_small.json`).

Per the spec's narrowing rule (the first test fails on SAML-D), the README claim narrows to
**same-generator public data**, and Q4 adds that it must be plentiful. With a frozen backbone and DP on the head only, this pipeline needs
public data from the same generator as the private data.

## Q2. head_dp_eps8 under public-only tuning against A4's 0.558

| | A4 (val-selected, seeded) | Stage H (fixed rounds, secure) |
|---|---|---|
| zero_shot | 0.5528 | 0.5528 |
| head_nodp | 0.5634 | 0.5514 ± 0.0000 |
| head_dp ε=1 | 0.5571 | 0.5530 ± 0.0001 |
| head_dp ε=3 | 0.5582 | 0.5485 ± 0.0006 |
| head_dp ε=8 | 0.5584 | 0.5497 ± 0.0003 |

Moving model selection inside the privacy budget costs 0.009 at ε=8. The Stage H DP models sit at
or just below the pretrained model (zero_shot 0.5528). For the ε=8 runs, A4-style validation selection
would have picked round 0, the pretrained model, for 2 of 3 fine-tuning seeds (round 10 for the third). On this backbone, private training adds
nothing measurable. The DP result amounts to "DP costs nothing on top of a good public model".

## Q3. Spread across pretraining seeds (LI-Small backbone)

| Pretraining seed | Pretrain test PR-AUC (LI-Small) | zero_shot | head_nodp | head_dp ε=1 | ε=3 | ε=8 | F1 at ε=8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.242 | 0.5528 | 0.5514 | 0.5530 | 0.5485 | 0.5497 | 0.610 |
| 1 | 0.227 | 0.5149 | 0.5217 | 0.5098 | 0.5243 | 0.5269 | 0.575 |
| 2 | 0.244 | 0.5161 | 0.5226 | 0.5151 | 0.5179 | 0.5191 | 0.595 |
| mean ± std | | 0.528 ± 0.018 | 0.532 ± 0.014 | | | **0.532 ± 0.013** | **0.593 ± 0.015** |

The spread across pretraining seeds (about 0.013) is 40× the spread across fine-tuning seeds (0.0003).
The seed-0 backbone, the one used in A4, is the best of the three. Seeds 1 and 2 used the settings tuned
for seed 0 (spec decision, cost: slightly pessimistic rows). On those backbones private training adds
+0.003 to +0.012 at ε=8.

## Q4. Gain from private data at 100% / 10% / 1% public data

Gain is head_dp_eps8 − zero_shot and head_nodp − zero_shot (mean ± std over 3 fine-tuning seeds).
full_nodp (whole model, no DP, A4's settings and validation selection) is an optimistic no-DP ceiling.

| Public data | zero_shot | head_dp_eps8 − zero_shot | head_nodp − zero_shot | full_nodp − zero_shot |
|---|---|---|---|---|
| 100% (seed 0) | 0.5528 | −0.0031 ± 0.0003 | −0.0014 ± 0.0000 | +0.0124 (A4) |
| 100% (seeds 1, 2) | 0.515 | +0.012, +0.003 | +0.007, +0.007 | — |
| 10% | 0.1367 | **+0.0266 ± 0.0015** | +0.0010 ± 0.0003 | +0.3375 |
| 1% | 0.0964 | **−0.0561 ± 0.0229** | +0.0133 ± 0.0013 | +0.2621 |

- **With a strong backbone the private data adds almost nothing,** with or without DP.
- **With a weaker backbone the private data matters a lot,** but only when the whole model trains:
  full fine-tuning gains +0.26 to +0.34, while head-only training gains at most +0.03.
- **At 1% the tuned DP settings hurt.** They were tuned for the strong backbone (lr 3e-3, 10 rounds),
  and at ε=8 they push the head below the pretrained model, with large variance across seeds (0.040 ±
  0.023). ε=1 (1 round, lr 3e-4) stays near zero_shot. Settings tuned for one backbone do not carry over
  to a much weaker one.

This points to DP fine-tuning of more than the head as the next lever, which the spec excludes from
Stage H.

## Q5. Flower + SecAgg+ with head DP vs in-process

`flower_head_dp_eps8.json`: LI-Small seed-0 backbone, account DP ε=8 on the whole head, tuned settings
(lr 3e-3, 10 rounds), SecAgg+ over 10 banks, secure DP randomness.

- **Gate 1, aggregation:** all 10 rounds' secure aggregates are within 2.96e-6 to 3.65e-6 of the plain
  weighted average, against the bound n·2c/Q + 1e-6 = 7.73e-5. **Pass.**
- **Gate 2, utility:** pooled test PR-AUC 0.5494 vs 0.5495 in-process (same backbone, ε=8, seed 0); the
  gap is 0.0001 against an allowed 0.005. **Pass.**
- **Privacy:** every bank spent ε ≤ 8 (7.990–7.9997), δ = 1e-6.
- **Cost:** 38–57 s per round.

The run wrote each bank's local model to disk (`--debug-dump-locals`) to check gate 1, which defeats
SecAgg+'s purpose. The dumps and the cached per-bank embeddings (derived from private data) were deleted
after the check.

## Q6. A0 baselines with three seeds and tuned FedProx

`a0_three_seeds.json`. Seed 0 is from `results/comparison/ibm_hi_small.json` (A0); seeds 1–2 are from
this stage. Same arguments, 20,000-account cap, no DP. Std is the population std over 3 seeds.

| Model | PR-AUC | F1 | Recall @ 1% FPR | Per-seed PR-AUC | Train time (s) |
|---|---|---|---|---|---|
| centralized | 0.527 ± 0.005 | 0.583 ± 0.011 | 0.706 | 0.523, 0.535, 0.523 | ~7,600–8,200 |
| gbdt | 0.519 ± 0.011 | 0.547 ± 0.013 | 0.706 | 0.525, 0.529, 0.503 | 80–96 |
| fedavg | 0.392 ± 0.044 | 0.453 ± 0.060 | 0.613 | 0.336, 0.443, 0.397 | ~6,800–6,900 |

A0's single-seed FedAvg (0.336) was its lowest seed. FedAvg varies 4–9× more across seeds than the
centralized models.

FedProx μ sweep (seed 0). μ is chosen on HI-Small validation; this row has no DP, so that is allowed.

| μ | Val PR-AUC | Test PR-AUC | F1 | Recall @ 1% FPR |
|---|---|---|---|---|
| 0.1 (A0) | not recorded | 0.203 | 0.259 | 0.451 |
| 0.01 | 0.180 | 0.280 | 0.351 | 0.584 |
| **0.001** | **0.203** | **0.368** | **0.441** | 0.590 |

Tuned FedProx (μ=0.001) is within FedAvg's seed range and beats FedAvg's seed 0 (0.336). A0's
conclusion that FedProx was worse than FedAvg came from an untuned μ. The A0 run did not store a
validation PR-AUC for μ=0.1, but its test score is far below the others.

## Alert budgets

When a bank can review only the top-k scored transactions (pooled test, 1,798 laundering in total):

| Model | Precision @ 1,000 | Recall @ 1,000 | Precision @ 5,000 | Recall @ 5,000 |
|---|---|---|---|---|
| head_dp_eps8, LI-Small seed 0 | 0.853 | 0.475 | 0.236 | 0.656 |
| head_dp_eps8, backbone seeds 1 / 2 | 0.801 / 0.825 | 0.446 / 0.459 | 0.238 / 0.227 | 0.661 / 0.632 |
| centralized (A0, seeds 1–2) | 0.815 / 0.796 | 0.453 / 0.443 | 0.231 / 0.237 | 0.642 / 0.660 |
| gbdt (A0, seeds 1–2) | 0.783 / 0.733 | 0.435 / 0.408 | 0.227 / 0.224 | 0.631 / 0.624 |
| fedavg (A0, seeds 1–2) | 0.727 / 0.626 | 0.404 / 0.348 | 0.208 / 0.194 | 0.577 / 0.540 |
| head_dp_eps8, LI-Small 10% | 0.379 | 0.211 | 0.128 | 0.355 |

At 1,000 alerts (0.1% of the test set) the private model's alerts are about 85% laundering and catch
about 47% of all laundering. At 5,000 alerts, recall rises to about 66% and precision falls to about 24%.
A0 seed 0 predates these metrics and is not back-filled.

## Client 0

`client0.json`, scored with head_dp_eps8 (LI-Small seed-0 backbone, seed 0):

- Client 0 has 507,835 rows from 2,978 banks. One bank (raw code "070", built code 30527) is **88.6%** of
  them.
- That bank has only **15 sending accounts**, each sending about 30,000 transactions.
- Its laundering uses Cheque, Credit Card and Cash, like its legitimate traffic. Everywhere else in
  HI-Small, 98.7% of laundering is ACH.
- The model judges a transaction by the sender's recent history, which at this bank looks the same for
  laundering and legitimate payments. Test PR-AUC inside the bank is **0.0014**, against a base rate of
  0.0013: chance.
- The client's laundering rate is unremarkable (0.13–0.14% in every split). The problem is
  separability, not label scarcity.

Sender-history features cannot fix this. Receiver and graph features can (where the money goes, and
whether it fans out), which is Stage A2's job.

## Published HI-Small results

Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs", AAAI 2024, Table 2,
and Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models",
NeurIPS 2023 Datasets & Benchmarks, Table 2. Metric: minority-class (laundering) F1 in %, HI-Small,
60/20/20 temporal split, transaction (edge) classification, centralized, no privacy, mean ± std over
several runs. Rows below are as printed in those tables (the boosting rows originate in Altman et al.;
Egressy et al. report them alongside the GNNs).

| Model | Privacy | Graph features | HI-Small F1 (%) |
|---|---|---|---|
| Multi-PNA + EU | none, centralized | yes | 68.16 ± 2.65 |
| Multi-GIN + EU | none, centralized | yes | 64.79 ± 1.22 |
| Multi-PNA | none, centralized | yes | 64.59 ± 3.60 |
| XGBoost + graph features | none, centralized | yes | 63.23 ± 0.17 |
| LightGBM + graph features | none, centralized | yes | 62.86 ± 0.25 |
| **Ours: head_dp ε=8, LI-Small seed-0 backbone** | account ε=8, δ=1e-6, 10 banks, SecAgg+-compatible | no | **61.04** (3 fine-tuning seeds) |
| **Ours: same, mean over 3 pretraining seeds** | same | no | **59.3 ± 1.5** |
| Ours: A0 centralized, no pretraining | none, centralized | no | 58.3 ± 1.1 |
| Multi-GIN | none, centralized | yes | 57.15 ± 4.99 |
| PNA | none, centralized | yes | 56.77 ± 2.41 |
| GIN + EU | none, centralized | yes | 47.73 ± 7.86 |
| GIN | none, centralized | yes | 28.70 ± 1.13 |

How comparable these are:
- **Same:** dataset, split proportions and direction, and metric.
- **Different:** they train centrally with no privacy, and they use graph features or message passing
  over the whole transaction graph. We use per-account sender histories inside each bank, federated,
  with account-level DP. They do not pretrain on LI-Small.
- **Threshold:** neither paper states its decision rule; Altman's figures suggest a fixed 0.5. We pick
  the F1-maximising threshold on validation, which favours us somewhat.
- **Exact rows:** our split is by time per our builder, not their released split files, so the test
  transactions are not guaranteed identical.

On these terms, the private federated model sits about 2 points below the best boosting models with
graph features and about 7 below the best graph network (4 and 9 using the mean over pretraining
seeds), and above several centralized GNNs. Closing that
gap is what graph features (A2) are for.

## Provenance of every row

| File | Rows | dp_randomness | selection |
|---|---|---|---|
| `ft_li_small.json` | zero_shot, head_nodp, head_dp ε=1/3/8 | secure | fixed_rounds |
| `ft_li_seed1.json`, `ft_li_seed2.json` | same | secure | fixed_rounds |
| `ft_li_10pct.json`, `ft_li_1pct.json` | same | secure | fixed_rounds |
| `ft_li_10pct_full.json`, `ft_li_1pct_full.json` | full_nodp | secure (no DP rows) | val |
| `ft_saml_d.json` | zero_shot, head_nodp, head_dp ε=1/3/8 | secure | fixed_rounds |
| `ft_saml_d_full.json` | full_nodp | secure (no DP rows) | val |
| `ft_amlsim.json` | zero_shot, head_nodp, head_dp ε=8 | secure | fixed_rounds |
| `flower_head_dp_eps8.json` | head_dp ε=8 via Flower + SecAgg+ | secure | fixed_rounds |
| `a0_three_seeds.json`, `a0_fedprox_mu*.json` | no-DP baselines | n/a | val (best round) |
| `results/comparison/ibm_hi_small_a4.json` (A4, cited above) | all | seeded | val |

zero_shot is round 0 by definition. Every DP row spent ε ≤ its target on every bank (largest: 7.9997
at ε=8, 2.998 at ε=3, 0.999 at ε=1), δ = 1e-6.

## Run notes

- **Volume drop:** the external volume dropped at 03:43 on 2026-11-19 during batch 3, and `ft_amlsim`
  and `flower_head` failed at start (script not found). Only those two were rerun (`batch3b`); no
  finished run was repeated.
- **Fine-tuning settings** (`tuned_settings.json`, LI-Small backbone, tuned on LI-Medium):
  - no DP: lr 3e-3, last layer, R=10
  - ε=1: lr 3e-4, whole head, R=1
  - ε=3 and ε=8: lr 3e-3, whole head, R=10

  Several picks sit at the edge of the grid. The grid was not extended (spec).
- **Running time:** `ft_li_seed2` took 2 h 40 min, against 38 min for seed 1. Its numbers are in line
  with the other seeds, and the cause was not found (likely machine load).

## What the README now claims

1. Federated account-level DP at ε=8 keeps 97% of the best no-DP PR-AUC when the backbone is pretrained
   on **plentiful public data from the same generator** (full LI-Small). With 10% or 1% of it, the test
   fails (34%, 11%). With different-generator public data (SAML-D, AMLSim) the
   frozen backbone does not transfer and head DP is at chance.
2. On a strong backbone the gain comes from the public data. Private training, with or without DP, adds
   nothing measurable.
3. The DP model matches or beats centralized training without pretraining on the pre-registered backbone
   (0.550 vs 0.527). Averaged over pretraining seeds the margin (0.532 vs 0.527) is within the noise.
4. The same holds through Flower with SecAgg+ (0.5494 vs 0.5495), with secure DP noise and model
   selection inside the privacy budget. The F1 threshold is still chosen on HI-Small validation; PR-AUC
   does not depend on it.
