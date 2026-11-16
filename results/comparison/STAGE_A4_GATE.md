# Stage A4 gate: public pretraining on LI-Small, account-level DP on the head

Sources: `results/comparison/ibm_hi_small_a4.json` (A4) and `results/comparison/ibm_hi_small.json` (A0).
A4 command: `python3 -W ignore experiments/run_dp_finetune.py --ckpt checkpoints/pretrain_li_small.pt --epsilons 1 3 8 --seeds 0 1 2 --dp-train last_layer --lr 1e-3 --models zero_shot head_nodp head_dp`,
then the same with `--models full_nodp --seeds 0`, merged. Pretraining: `results/pretrain/li_small.json`.

## What changed from A0

1. **Public pretraining.** The full model is trained without DP on IBM LI-Small (public; 4,154,408 train
   transactions, 1,813 laundering, 675,527 accounts, 5 epochs, best epoch 3 by validation PR-AUC 0.161;
   96 min). It never sees HI-Small.
2. **DP on the head only.** The backbone is frozen and turns each HI-Small example into a 32-wide embedding once.
   Federated account-level DP-SGD then trains only the classifier's last layer (33 parameters) instead of 44,449.
3. **Every train account** (no 20,000 cap), **laundering rows first** inside each sampled account, and the
   starting model can be kept (round 0) if DP training does not improve validation.

## Results (pooled HI-Small test set, 1,015,882 transactions, 1,798 laundering; mean ± std over seeds 0–2)

A model that ranks at random scores PR-AUC ≈ 0.0018.

| Model | PR-AUC | Recall @ 1% FPR | F1 | Train time (s) | ε spent |
|---|---|---|---|---|---|
| A0 centralized (no privacy) | 0.523 | 0.700 | 0.591 | 8165 | none |
| A0 fedavg | 0.336 | 0.544 | 0.397 | 6900 | none |
| A0 fedprox | 0.203 | 0.451 | 0.259 | 7218 | none |
| A0 account_dp_eps1/3/8 | 0.001 | 0.004 | ≤ 0.001 | 3696–4905 | 0.99 / 2.99 / 7.99 |
| zero_shot (pretrained, no private training) | 0.553 | 0.740 | 0.612 | 110 (scoring) | none |
| head_nodp (FedAvg on the head, no DP) | 0.563 ± 0.000 | 0.749 ± 0.001 | 0.615 ± 0.000 | 175 | none |
| full_nodp (FedAvg on the whole pretrained model, seed 0) | 0.565 | 0.716 | 0.625 | 15430 (machine shared; estimated ~7000 unloaded from A0 fedavg) | none |
| head_dp_eps1.0 | 0.557 ± 0.000 | 0.739 ± 0.000 | 0.614 | 335 | max 1.00 (account-level) |
| head_dp_eps3.0 | 0.558 ± 0.000 | 0.739 | 0.614 ± 0.001 | 361 (median) | max 3.00 (account-level) |
| head_dp_eps8.0 | 0.558 ± 0.000 | 0.739 | 0.614 ± 0.001 | 345 (median) | max 8.00 (account-level) |

Train times for head runs exclude the one-off embedding pass (266 s for all 5,078,345 examples). Two DP
runs took 4,768 s and 16,712 s because another job saturated the machine; the medians are shown.

Per client (seed 0, PR-AUC):

| Client | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 centralized | 0.029 | 0.504 | 0.585 | 0.674 | 0.544 | 0.540 | 0.622 | 0.565 | 0.568 | 0.587 |
| A0 fedavg | 0.020 | 0.359 | 0.357 | 0.403 | 0.318 | 0.347 | 0.402 | 0.381 | 0.352 | 0.379 |
| zero_shot | 0.033 | 0.561 | 0.641 | 0.652 | 0.566 | 0.561 | 0.638 | 0.597 | 0.596 | 0.620 |
| head_nodp | 0.035 | 0.564 | 0.661 | 0.662 | 0.566 | 0.572 | 0.647 | 0.615 | 0.598 | 0.628 |
| full_nodp | 0.028 | 0.554 | 0.669 | 0.689 | 0.566 | 0.589 | 0.691 | 0.613 | 0.627 | 0.615 |
| head_dp_eps8.0 | 0.034 | 0.564 | 0.665 | 0.657 | 0.566 | 0.569 | 0.640 | 0.606 | 0.595 | 0.627 |

## Gate questions

**1. The gate.** Mean `head_dp_eps8.0` PR-AUC is 0.5584 ± 0.0002 (seeds: 0.5586, 0.5585, 0.5583). The bar is
half of the best no-DP federated PR-AUC, max(A0 fedavg 0.336, head_nodp 0.563, full_nodp 0.565) =
0.565 (full_nodp), so the bar is 0.283. **Passed**: DP at ε=8 keeps 98.8% of the best no-DP federated
PR-AUC, against under 1% in A0.

**2. What pretraining alone gives.** The pretrained model with no private training scores 0.553, above A0
FedAvg (0.336) and above A0's centralized model trained on HI-Small itself (0.523), and it beats that
centralized model on 9 of 10 clients. Training the head on HI-Small without DP adds 0.011 (0.563);
fine-tuning the whole model without DP (20,000 train accounts per client, as in A0) gives 0.565 (+0.012 vs zero_shot) but lower recall at 1% FPR (0.716 against 0.740). So the private data, used without any privacy, adds about 0.01 PR-AUC over the public pretraining.

**3. What the private data adds under DP.** Very little. `head_dp_eps8.0` minus `zero_shot` is +0.006 PR-AUC
and −0.002 recall at 1% FPR. Eight of nine DP runs select round 1 of 10 on validation (one ε=8 run selects
round 10); later rounds do not improve validation. In this setting almost all of the gain over A0 comes from
the public data, not from federated learning on the private data.

**4. The ε curve.** Flat: 0.5571 (ε=1), 0.5582 (ε=3), 0.5584 (ε=8). Recall at 1% FPR is 0.739 at every ε.
With 33 trained parameters the DP noise is no longer the limit; the frozen embedding is.

**5. Overlap between public and private data** (`experiments/check_public_overlap.py`): 0 shared transaction
rows (6,924,049 public, 5,078,345 private); 3 shared sender account IDs (681,283 public, 496,999 private).
The features are fixed transforms (logs, cyclic time, one-hot codes) with no scaler fitted on any data,
so no HI-Small statistic reaches the pretrained model.

**6. Configurations tried** (ε=8, seed 0, chosen on pooled validation PR-AUC; test not used):

| DP layers | lr | Validation PR-AUC | Selected round |
|---|---|---|---|
| classifier (1,089 params) | 1e-3 | 0.3952 | 0 |
| classifier (1,089 params) | 1e-2 | 0.3954 | 4 |
| **last layer (33 params)** | **1e-3** | **0.3962** | 1 |
| last layer (33 params) | 1e-2 | 0.3958 | 4 |

References at the same time: zero_shot 0.3952, head_nodp 0.3981 (lr 1e-3) / 0.3975 (lr 1e-2). The four DP
configurations differ by at most 0.001.

## Other findings

- Client 0 stays at PR-AUC 0.03 under every model, pretrained or not. Its laundering differs in kind.
- DP rows have F1 thresholds near 0.02–0.04 against 0.98 for zero_shot: the head's bias is reset to the
  HI-Small prior before training. A bias shift does not change the ranking, so PR-AUC is unaffected.

## Limitations

- **Same generator.** LI-Small and HI-Small come from the same IBM simulator, which is the favourable case
  for public pretraining. A real consortium would have public data that differs from its private data.
  The follow-up that bounds the unfavourable case is pretraining on AMLSim (`datasets_capstone/ibmamlsim`),
  a different simulator with account histories.
- **One pretraining seed.** The seed spread (± 0.000) covers only DP noise and sampling with a fixed
  backbone; variation from pretraining is not measured. full_nodp has one seed, like A0.
- Hyperparameter choice, round selection and the F1 threshold use validation data and are outside ε.
  Per-bank counts and the pooled train fraud rate (bias initialisation) are treated as public.
