# Stage B1 gate: Flower + SecAgg+

Sources: `results/flower/b1_fedprox.json` and `results/flower/b1_account_dp_eps8.json`, produced by

```bash
python3 -W ignore experiments/run_flower.py --rounds 10 --debug-dump-locals --out results/flower/b1_fedprox.json --max-train-accounts 20000
python3 -W ignore experiments/run_flower.py --rounds 10 --debug-dump-locals --account-dp-eps 8 --out results/flower/b1_account_dp_eps8.json --max-train-accounts 20000
```

Both ran on 2026-11-16/17, seed 0, IBM AML HI-Small, 10 clients, the same 20,000-account cap as A0.
`--debug-dump-locals` wrote each bank's local model to disk so the aggregate could be checked. That
defeats SecAgg+ and is for this check only. The dumps were deleted after the check.

## Gate: SecAgg+ aggregate equals the plain weighted average

Each round, the global model Flower produced under SecAgg+ was compared with the plain weighted
average (by train count) of the same ten local models. The bound is one quantization step per client,
n·2c/Q = 10 · 2·16 / 2²² = 7.63e-5, plus 1e-6 slack: **7.73e-5**.

| Round | FedProx max abs error | Account DP ε=8 max abs error |
|---|---|---|
| 1 | 8.58e-6 | 9.52e-6 |
| 2 | 8.68e-6 | 8.14e-6 |
| 3 | 9.11e-6 | 8.28e-6 |
| 4 | 8.79e-6 | 8.52e-6 |
| 5 | 9.00e-6 | 9.67e-6 |
| 6 | 8.90e-6 | 8.47e-6 |
| 7 | 8.52e-6 | 9.16e-6 |
| 8 | 9.31e-6 | 8.47e-6 |
| 9 | 9.00e-6 | 9.00e-6 |
| 10 | 8.68e-6 | 8.39e-6 |

**Passed.** The worst round is 9.7e-6, an eighth of the bound and about 1.3 quantization steps
(2c/Q = 7.6e-6). Rounding errors from different clients mostly cancel, so the observed error stays far
below the worst case of one full step per client.

## Seconds per round

| Run | Rounds 1–10 (s) | Total |
|---|---|---|
| FedProx | 731, 767, 666, 586, 650, 650, 586, 598, 630, 761 | 6,624 s |
| Account DP ε=8 | 619, 576, 585, 561, 508, 386, 400, 411, 597, 646 | 5,288 s |

Round 1 includes Ray and Flower start-up. Round times come from the file times of each round's
saved global model. Ray ran 2 actors with 1 CPU each, so at most two banks trained at once.
A0's sequential loop took 7,218 s for FedProx and 4,905 s for account DP ε=8. A0's DP run shared
the CPU with other jobs, so these times do not show what SecAgg+ costs; B3 measures that.

## Metrics beside A0 (pooled test set, 1,015,882 transactions, 1,798 laundering)

| Model | PR-AUC | ROC-AUC | Recall @ 1% FPR | F1 | ε spent |
|---|---|---|---|---|---|
| A0 `fedprox` | 0.203 | 0.874 | 0.451 | 0.259 | none |
| B1 FedProx under SecAgg+ | 0.209 | 0.865 | 0.448 | 0.276 | none |
| A0 `account_dp_eps8.0` | 0.0014 | 0.411 | 0.004 | 0.000 | max 7.99 (account-level) |
| B1 account DP ε=8 under SecAgg+ | 0.0014 | 0.390 | 0.006 | 0.002 | 7.99 per client (account-level) |

Per-client PR-AUC, A0 / B1 (FedProx):
0: 0.016/0.020, 1: 0.208/0.209, 2: 0.255/0.259, 3: 0.291/0.305, 4: 0.231/0.242,
5: 0.201/0.207, 6: 0.251/0.259, 7: 0.165/0.172, 8: 0.191/0.199, 9: 0.226/0.223.

The two runs match closely but not exactly, as expected. A0 uses one random generator across all
clients in turn. B1 gives each client its own generator for each round, so batches and dropout masks
differ. A0 and B1 also start from slightly different output biases: B1 sets the prior from the pooled
train rate in the metadata, while A0 uses the capped data's rate. The gate is the per-round equality
above, not these metrics. B1 shows that running the pipeline under Flower + SecAgg+ does not change
what it learns. Account DP still fails without public pretraining, as in A0. Stage A4 is the fix for
that.

Account DP privacy (from the JSON): σ = 0.735, sampling rate 0.0128, 1,580 steps, 20,000 accounts per
client, δ = 1e-6, ε = 7.99 for every client.

## SecAgg+ parameters

| Parameter | Value |
|---|---|
| Protocol | Flower `SecAggPlusWorkflow` |
| `num_shares` | 10 (every client) |
| `reconstruction_threshold` | 7 (min(⌈0.7n⌉, n−1)) |
| `clipping_range` | 16.0; a client refuses to send any trainable value outside it or any non-finite value |
| `quantization_range` | 2²² |
| `modulus_range` | 2³² |
| `max_weight` | largest client train count, so the weighted sum cannot overflow |
| Failures | the server refuses to aggregate if any client fails; every round's aggregate must exist |

## Out of scope

- Client dropouts mid-round: B1 stops the run on any failure. The B2 gate tests a bank pod killed mid-round.
- Client-level DP under Flower: only A0's v1 baseline has it.
- Message sizes and SecAgg+ overhead: B3.
- **Seeded noise.** DP noise comes from a seeded PyTorch generator so runs reproduce. A real deployment
  needs a cryptographically secure source. B2 fixes this.
- One seed only, the same as A0.
