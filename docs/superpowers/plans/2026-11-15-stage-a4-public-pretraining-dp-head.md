# Stage A4: Public Pretraining + Head-Only Account-Level DP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make account-level DP at ε=8 keep at least half of the no-DP PR-AUC on IBM HI-Small, by pretraining the model without DP on public IBM LI-Small and fine-tuning only its classifier head under DP.

**Architecture:** LI-Small is built with the A0 builder as one client and checked for shared accounts with HI-Small. A centralized pretraining run on LI-Small saves a checkpoint. For HI-Small, the frozen pretrained backbone turns every example into its 32-dimensional `embeddings` vector once (eval mode); federated training then runs on those vectors with a model that is only the classifier (1,089 parameters), using A0's `run_federated` and account-level DP-SGD unchanged except for fraud-rows-first sampling inside each account. A small validation probe picks how much of the head trains under DP (all 1,089 parameters or only the last layer's 33) and the learning rate; those are then fixed for every ε.

**Tech Stack:** Python 3.12, torch 2.8 (`torch.func`), opacus 1.5.4 (accounting only), pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md` (sections "If A0's privacy cost is too high" with its 2026-11-15 result, roadmap row A4, "Assumptions the paper must state").

## Verified before writing (2026-11-15)

- LI-Small's header is identical to HI-Small's. It has 6,924,049 rows and 3,565 laundering rows (0.0515%); HI-Small has 5,078,345 and 5,177.
- The model's `embeddings` output is 32-wide; the classifier is `Linear(32,32) → GELU → Dropout → Linear(32,1) → Sigmoid` (1,089 parameters; the last layer is 33). `classifier(embeddings) == fraud_prob` to 1e-6.
- In eval mode an example's embedding does not depend on the other rows of its batch (LayerNorm only, no BatchNorm). Account-level DP on embeddings needs exactly this.
- HI-Small build peaked at 1.6 GB RSS; LI-Small is 1.36× the rows.

## Why these levers

In the regime A0 ran in, the signal-to-noise of a whole DP run is roughly `F · ε · √K / √d`: F fraud accounts per bank, K banks averaged by the server, d trained parameters. Batch size, steps and epochs mostly cancel (measured: 1.2–1.8× from 20× larger batches). So the plan moves the three things that matter: d (44,449 → 1,089 or 33), F (no account cap, fraud rows always sampled), and the starting point (a pretrained model has less to learn). Learning rate and rounds are tuned on validation because they do not cancel under Adam.

## Global Constraints

- Runs on an Apple M1 with 8 GB RAM, CPU only. No step may hold all sequences as a dense tensor; peak memory of a data build stays under 6 GB.
- No new dependencies.
- No SMOTE or any synthetic rows. No statistic fitted on the data enters a feature (no scaler, no learned vocabulary).
- Splits are chronological 60/20/20 with global time cut-offs and are used exactly as built. Nothing re-splits them.
- Privacy unit is the sending account. δ = 1e-6, and a run aborts if δ ≥ 1 / (train accounts of any client).
- Model selection uses validation data only. Test data is read once per run, for the selected round.
- Headline metrics are PR-AUC and recall at 1% false-positive rate.
- Existing files `run_training.py`, `experiments/run_model_comparison.py`, `experiments/run_epsilon_sweep.py`, `backend/`, `frontend/` and every `src/models/*.py` are not modified. No state-dict key changes.
- Raw CSVs live under `datasets_capstone/ibm_aml/` (already git-ignored) and are never committed. `*.pt` files are never committed.
- LI-Small is public pretraining data and sits outside ε (spec assumption). HI-Small is the only protected data.
- DP fine-tuning trains the classifier head or its last layer only; DP runs on HI-Small use every account (no `--max-train-accounts`). The one capped run is `full_nodp`, at A0's cap of 20,000 so it is comparable to A0's `fedavg`.
- Hyperparameters for DP (trained layers, learning rate) are chosen once, on pooled validation PR-AUC at ε=8, and reused for every ε and seed. The spec already counts tuning as outside ε; the gate file must list every configuration tried.
- The gate: mean over seeds of `head_dp_eps8.0` PR-AUC ≥ half of the best no-DP federated PR-AUC, where "best" is the maximum of A0 `fedavg`, `head_nodp` and `full_nodp`.
- Long runs write their log and JSON to `~/capstone_runs/` on the internal disk and are copied into the repo after they exit (the external drive remounts; see the A0 ledger).
- All commands run from the repo root. Tests run with `python3 -m pytest tests -q`. Commits are local only; nothing is pushed.
- Work on branch `stage-a0-core-pipeline`.

## Review Focus

1. A checkpoint from a different feature count or config is loaded for fine-tuning → load must fail with a clear error, never a silent partial load (strict `load_state_dict`, `num_features` checked against the HI metadata).
2. LI-Small and HI-Small share transactions → the overlap check must stop the stage before pretraining, not just print a number.
3. Embeddings computed with dropout active → two calls on the same rows must give identical vectors (backbone in eval mode, no grad).
4. An account with more fraud rows than `examples_per_account` under fraud-rows-first → it contributes exactly `examples_per_account` rows, all fraud; no account ever contributes more rows than the cap.
5. An example's embedding changes with its batch neighbours (a future BatchNorm, train mode) → the per-account sensitivity bound is void; a test must compare batch-of-1 with batch-of-many.
6. DP fine-tuning makes a good pretrained model worse → the untouched starting model must stay selectable (round 0), and the selected round is reported.
7. The DP head run accounts with the uncapped account count of each client and aborts on δ before any training; backbone weights are bit-identical before and after fine-tuning.

---

### Task 1: LI-Small build and overlap check

**Files:**
- Create: `experiments/check_public_overlap.py`, `tests/test_public_overlap.py`
- Generated (not committed except metadata): `datasets_capstone/ibm_aml/li_small/LI-Small_Trans.csv` (extracted from `archive.zip`), `data/ibm_li_small/` via `experiments/build_ibm_aml.py --src ... --out data/ibm_li_small --clients 1`
- Modify: `.gitignore` only if `data/ibm_li_small/metadata/dataset_info.json` is not already tracked-by-pattern (A0's HI metadata is tracked; keep the same treatment)

**Interfaces:**
- Produces: `overlap_report(public_csv: Path, private_csv: Path) -> dict` with keys `shared_accounts` (int, (bank, account) pairs present as a sender in both), `public_accounts`, `private_accounts`, `shared_rows` (int, rows equal on all 11 raw columns), `public_rows`, `private_rows`; CLI `python3 experiments/check_public_overlap.py PUBLIC.csv PRIVATE.csv [--max-shared-row-frac 0.001]` prints the report and exits 1 when `shared_rows / public_rows > max-shared-row-frac`.

- [ ] **Step 1: Write the failing tests** in `tests/test_public_overlap.py` using `tests/helpers.py` `make_raw`/`write_raw`:
  - `test_disjoint_files_report_zero_overlap`: two raw frames with disjoint account strings → `shared_rows == 0`, `shared_accounts == 0`.
  - `test_copied_rows_are_counted`: private frame = public frame plus 5 extra rows → `shared_rows == len(public)`.
  - `test_cli_exits_nonzero_on_heavy_overlap`: same files as above, `main([...])` raises `SystemExit` with code 1.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_public_overlap.py -q` — Expected: 3 errors (module missing).
- [ ] **Step 3: Implement** `overlap_report` and `main(argv)` in `experiments/check_public_overlap.py`. Read with the same dtypes as `src/data/ibm_aml.load_transactions` (bank codes as strings, zero padding kept); sender key = `From Bank + "|" + Account`; row equality via a merge on all 11 raw columns.
- [ ] **Step 4: Run** the test file — Expected: 3 passed. Run `python3 -m pytest tests -q` — Expected: all pass.
- [ ] **Step 5: Extract and check the real files.** `unzip -j archive.zip LI-Small_Trans.csv -d /Volumes/Untitled/datasets_capstone/ibm_aml/li_small/`, then `python3 experiments/check_public_overlap.py <LI> <HI>`. Expected: exit 0. Record all six counts in the ledger. Exit 1 stops the stage: report to the user.
- [ ] **Step 6: Build** `/usr/bin/time -l python3 experiments/build_ibm_aml.py --src <LI> --out data/ibm_li_small --clients 1`. Expected: one client folder, peak RSS under 6,000,000,000 bytes, printed split table. Record banks, rows, fraud per split.
- [ ] **Step 7: Commit** `experiments/check_public_overlap.py tests/test_public_overlap.py data/ibm_li_small/metadata` — message `Add public/private overlap check and build LI-Small`.

### Task 2: Fraud rows first inside each sampled account

**Files:**
- Modify: `src/federated/dp_sgd.py` (`AccountDPConfig`, `_sample_rows`, `train_local_account_dp`)
- Test: `tests/test_dp_sgd.py`

**Interfaces:**
- Produces: `AccountDPConfig.fraud_rows_first: bool = False` (default keeps A0 behaviour bit-identical); `_sample_rows(..., per_account, generator, labels=None)` where `labels` is the example-label tensor aligned with the dataset's example positions; when given, each picked account's rows are ordered fraud first (random order within fraud, then random order within legitimate) before taking `min(count, per_account)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_fraud_rows_first_always_includes_the_fraud_row`: one account with 10 rows, row 7 fraud, `per_account=4`, sample rate 1.0, 200 draws → row 7 picked in all 200 with `labels`, in fewer than 150 without.
  - `test_fraud_rows_first_caps_rows_per_account`: one account with 6 fraud rows of 8, `per_account=4` → exactly 4 rows, all fraud.
  - `test_default_sampling_unchanged`: with `labels=None` the sampled rows equal those from a fixed-seed call recorded before the change (compute the golden list once in RED and paste it).
  - `test_train_local_account_dp_fraud_first_runs`: `tiny_client` train set, `AccountDPConfig(..., fraud_rows_first=True)`, 3 steps → returns finite loss.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_dp_sgd.py -q` — Expected: the 4 new tests fail (unexpected keyword / attribute).
- [ ] **Step 3: Implement.** In `_sample_rows`, per picked account build the within-group order as `randperm` then stable sort by `-labels[group rows]`; take the first `take` positions. `train_local_account_dp` passes `dataset.example_labels` only when `cfg.fraud_rows_first`. Add a one-line comment: the selection depends only on the account's own rows, and the account's whole contribution is clipped, so it costs no ε.
- [ ] **Step 4: Run** the file — Expected: all pass. Then `python3 -m pytest tests -q` — Expected: all pass, including A0's golden test `test_local_fit_extraction_preserves_federated_results` unchanged.
- [ ] **Step 5: Commit** `src/federated/dp_sgd.py tests/test_dp_sgd.py` — `Account DP: optional fraud-rows-first sampling inside each account`.

### Task 3: Pretraining on the public set

**Files:**
- Create: `experiments/pretrain_public.py`, `tests/test_pretrain_public.py`
- Generated: `checkpoints/pretrain_li_small.pt` (ignored), `results/pretrain/li_small.json`
- Modify: `.gitignore` (add `!results/pretrain/*.json`)

**Interfaces:**
- Consumes from `experiments/run_ibm_comparison.py`: `parse_args`, `model_builder`, `BatchIter`, `score`, `labels_of`, `snapshot`, `is_finite`, `DEFAULT_CONFIG`; `train_local` from `run_training`; `fraud_metrics` from `src/utils/metrics`.
- Produces: `pretrain(data_path: Path, epochs: int, seed: int, args) -> dict` returning `{"state_dict", "config", "num_features", "val_pr_auc_by_epoch", "best_epoch", "test": fraud_metrics-dict, "train_seconds"}`; CLI `python3 experiments/pretrain_public.py [--data-path data/ibm_li_small] [--epochs 5] [--seed 0] [--max-train-accounts N] [--out-ckpt checkpoints/pretrain_li_small.pt] [--out-json results/pretrain/li_small.json]`. The checkpoint is `torch.save({"state_dict", "config", "num_features", "source"})`; the JSON holds everything except `state_dict`.

- [ ] **Step 1: Write the failing tests** (data: `tiny_client` fixture's build dir with `n_clients=1`; build it in the test with `build_dataset(..., n_clients=1)`):
  - `test_pretrain_returns_best_epoch_by_val`: `epochs=2` → `len(val_pr_auc_by_epoch) == 2`, `best_epoch == argmax + 1`, `state_dict` keys equal `TemporalGraphTransformer(config).state_dict()` keys.
  - `test_cli_writes_checkpoint_and_json`: CLI on tmp paths → both files exist; JSON has no `state_dict`; checkpoint `num_features == 35`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_pretrain_public.py -q` — Expected: errors (module missing).
- [ ] **Step 3: Implement.** Centralized training on client 0 of the public set: Adam(lr, weight_decay 1e-5), `train_local(..., epochs=1)` per epoch with `BatchIter(..., args.oversample_ratio)`, score val after each epoch, keep the best finite snapshot, score test once with the best. Prior for the bias from the public set's train fraud rate.
- [ ] **Step 4: Run** the file and the suite — Expected: all pass.
- [ ] **Step 5: Run for real** (internal disk for log/JSON, then copy): `python3 -W ignore experiments/pretrain_public.py --epochs 5 --out-json ~/capstone_runs/li_small.json > ~/capstone_runs/pretrain.log 2>&1`. Expected: finishes; best LI val PR-AUC well above the LI val fraud rate. If one epoch takes over 30 min, rerun with `--max-train-accounts` and record the ruling. If `best_epoch` is the last epoch, run 3 more epochs (`--epochs 8`, same seed) and keep the better checkpoint: the backbone is fixed for the rest of the stage, so an under-trained one caps every later number.
- [ ] **Step 6: Commit** `experiments/pretrain_public.py tests/test_pretrain_public.py .gitignore results/pretrain/li_small.json` — `Pretrain the model on public IBM LI-Small`.

### Task 4: Head-only federated DP fine-tuning on embeddings

**Files:**
- Create: `experiments/run_dp_finetune.py`, `tests/test_run_dp_finetune.py`
- Modify: `experiments/run_ibm_comparison.py`: `aggregate` skips the contrastive-buffer copy when the model has no `contrastive_head`; `run_federated` gains `include_initial: bool = False` (when true, the starting global model is validation-scored as round 0 and can be selected) and its metrics dict gains `selected_round` (0 = the initial model, else 1..rounds). Defaults leave A0 results bit-identical.

**Interfaces:**
- Consumes: `run_federated`, `select_and_test`, `final_metrics`, `evaluate_split`, `load_indexed_client`, Task 2's `fraud_rows_first`, Task 3's checkpoint format.
- Produces:
  - `load_pretrained(path: Path, num_features: int) -> TemporalGraphTransformer` — raises `ValueError` naming both counts when the checkpoint's `num_features` differs; `load_state_dict(strict=True)`.
  - `class EmbeddingDataset` with `gather(idx) -> (emb[idx], labels[idx])`, `__len__`, `example_labels`, `example_accounts` (same contract as `SequenceIndexDataset`).
  - `embed(backbone, ds) -> torch.Tensor (len(ds), 32)` — eval mode, `torch.no_grad`, batches of `SCORE_BATCH`, returns `outputs["embeddings"]`.
  - `class HeadModel(nn.Module)` holding a deep copy of `backbone.classifier`; `forward(x, labels=None) -> {"fraud_prob": clamp(classifier(x).squeeze(-1), 1e-7, 1-1e-7)}`; `reset_bias(prior)` sets `classifier[3].bias` to the prior's log-odds; `set_trainable(which)` with `which in {"classifier", "last_layer"}` sets `requires_grad` (`last_layer` = `classifier[3]` only, 33 parameters).
  - CLI `python3 experiments/run_dp_finetune.py --ckpt checkpoints/pretrain_li_small.pt [--data-path data/ibm_hi_small] [--epsilons 1 3 8] [--seeds 0] [--mu 0] [--dp-train classifier|last_layer] [--models zero_shot head_nodp full_nodp head_dp] [--out results/comparison/ibm_hi_small_a4.json]` plus all A0 runner flags (it reuses `parse_args`; `fraud_rows_first=True`; `max_train_accounts` is forced to `None` for every model except `full_nodp`, which uses 20,000). Results keys: `zero_shot` (pretrained model as is, never trained on HI), `head_nodp` (FedAvg on the head, no DP), `full_nodp` (FedAvg on the whole pretrained model, no DP: the no-DP ceiling with the same public data), `head_dp_eps{e}` for each ε. Head runs call `run_federated(..., include_initial=True)`. Embeddings are computed once and shared by all head models and seeds. Same JSON layout as A0 so `experiments/summarize_ibm_results.py` reads it.

- [ ] **Step 1: Write the failing tests** (`tiny3` fixture; a checkpoint made by Task 3's `pretrain` on a 1-client tiny build):
  - `test_load_pretrained_rejects_feature_mismatch`: checkpoint with `num_features=35`, call with 34 → `ValueError` whose message contains `35` and `34`.
  - `test_embed_is_deterministic`: two `embed` calls on the same dataset → `torch.equal`.
  - `test_head_model_matches_full_model_scores`: for 16 examples, `HeadModel(backbone)(embed(...))["fraud_prob"]` equals `backbone(seqs)["fraud_prob"]` within 1e-6 (both eval mode, before `reset_bias`).
  - `test_backbone_unchanged_by_finetune`: run the CLI's `main` with `--rounds 1 --epsilons 8 --delta 1e-3` on `tiny3` → the loaded checkpoint's backbone tensors are bit-identical to the file's afterwards, and `results` has keys `zero_shot`, `head_nodp`, `full_nodp`, `head_dp_eps8.0`.
  - `test_embedding_independent_of_batch`: `embed` of one example alone equals its row from a batch of 16 within 1e-6.
  - `test_last_layer_trains_only_33_parameters`: `HeadModel.set_trainable("last_layer")` → trainable count 33; after a DP round `classifier[0]` is bit-identical.
  - `test_initial_model_can_be_selected`: `run_federated(..., include_initial=True)` with a stub whose rounds make validation worse → `selected_round == 0` and test metrics equal the initial model's; with the default `False`, A0's golden values are unchanged.
  - `test_dp_accounting_uses_uncapped_accounts`: with `--max-train-accounts 5` on the command line, every `privacy[c]["n_accounts"]` equals the client's full train-account count; `--delta 0.5` raises `SystemExit` before any training (patch `run_federated` to fail if called).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_run_dp_finetune.py -q` — Expected: errors (module missing).
- [ ] **Step 3: Implement** in `experiments/run_dp_finetune.py`. Per client, replace each split's dataset by `EmbeddingDataset(embed(backbone, ds), ds)`; `build_model` returns a fresh `HeadModel` from the loaded backbone with `reset_bias(pooled HI train fraud rate)`; reuse `run_federated(args, clients, build_model, g, mu=args.mu, account_dp_eps=e)` and `select_and_test`. The CLI logs the trained parameter count (1,089 or 33 for the default config). `full_nodp` builds the whole pretrained model (`load_pretrained` + `freeze_non_dp_heads`) and runs `run_federated` on the sequence datasets at the 20,000 cap. Guard in `aggregate`: `if hasattr(m, "contrastive_head")`.
- [ ] **Step 4: Run** the file and `python3 -m pytest tests -q` — Expected: all pass, A0 golden test unchanged.
- [ ] **Step 5: Validation probe, ε=8 only, seed 0** (each run is minutes: training is on 32-wide vectors). Run `--models zero_shot head_nodp head_dp --epsilons 8` for the grid `--dp-train {classifier, last_layer}` × `--lr {1e-3, 1e-2}` (4 runs, outputs `~/capstone_runs/a4_probe_*.json`). Record for each: pooled **validation** PR-AUC of the selected round, `selected_round`, and the score std. Pick the configuration with the best validation PR-AUC; test numbers are not looked at here. Expected: `head_nodp` ≥ `zero_shot` on validation, and at least one DP configuration with `selected_round > 0`.
  If every DP configuration selects round 0 (DP training never helps), try in this order and record each: (a) `--rounds 30`; (b) standardize embeddings with mean/std computed on **LI-Small** train embeddings (public, so free) — add `--standardize-from-public` only if reached; (c) `--dp-batch-accounts 1024`. If still round 0, stop and report to the user: the honest result is then "at ε ≤ 8 the private data adds nothing beyond public pretraining", and whether to publish that or change the design is their call.
- [ ] **Step 6: Commit** `experiments/run_dp_finetune.py experiments/run_ibm_comparison.py tests/test_run_dp_finetune.py` — `Head-only federated DP fine-tuning on pretrained embeddings`.

### Task 5: Full run and the A4 gate

**Files:**
- Create: `results/comparison/ibm_hi_small_a4.json`, `results/comparison/STAGE_A4_GATE.md`
- Modify: `README.md` (IBM AML results section: add the A4 rows), `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md` (A4 row status)

- [ ] **Step 1: Run** `python3 -W ignore experiments/run_dp_finetune.py --ckpt checkpoints/pretrain_li_small.pt --epsilons 1 3 8 --seeds 0 1 2 --dp-train <chosen> --lr <chosen> --models zero_shot head_nodp head_dp --out ~/capstone_runs/ibm_hi_small_a4.json > ~/capstone_runs/a4.log 2>&1` detached (`setsid` via perl, `caffeinate -ims -w`), then the same command with `--models full_nodp --seeds 0 --out ~/capstone_runs/ibm_hi_small_a4_full_nodp.json` (about 2 h; one seed, like A0). Copy the files into `results/comparison/` and `logs/` and merge `full_nodp` into the main JSON. Expected: 6 result keys, each with non-null `pr_auc`; three seeds for the head rows; `privacy` per client with `epsilon_spent` ≤ target.
- [ ] **Step 2: Write `STAGE_A4_GATE.md`:** table from `python3 experiments/summarize_ibm_results.py results/comparison/ibm_hi_small_a4.json` next to A0's `fedavg`, `fedprox`, `account_dp_eps*` rows; then answer, with numbers: (1) the gate as defined in Global Constraints, with mean ± std over seeds; (2) what pretraining alone gives (`zero_shot`, `head_nodp`, `full_nodp` vs A0 `fedavg`); (3) what the private data adds under DP: `head_dp_eps*` minus `zero_shot`, with the selected rounds — if this is zero, say so plainly; (4) the ε curve 1→3→8; (5) the overlap counts from Task 1; (6) every probe configuration tried in Task 4 Step 5 and its validation score. Then a Limitations paragraph: LI-Small comes from the same generator as HI-Small, which is the favourable case for public pretraining and must be stated as such; an out-of-distribution public source (the AMLSim files under `datasets_capstone/ibmamlsim`, a different simulator with account histories) is the follow-up that would bound the unfavourable case; tuning and round selection are outside ε; one pretraining seed.
- [ ] **Step 3: Update** the README results section and the spec's A4 row with the outcome.
- [ ] **Step 4: Commit** (A0's `!results/comparison/ibm_hi_small*.json` rule already covers the JSON) `README.md docs/superpowers/specs results/comparison/ibm_hi_small_a4.json results/comparison/STAGE_A4_GATE.md` — `Record Stage A4 results: public pretraining + head-only account DP`.
