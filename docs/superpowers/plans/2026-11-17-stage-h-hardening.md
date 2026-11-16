# Stage H Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v2 results survive review: out-of-distribution public pretraining (SAML-D, AMLSim), public-only tuning with fixed rounds, secure DP randomness, error bars, DP + SecAgg+ together, and the smaller fixes, all reported in one gate file.

**Architecture:** Code changes first (Tasks 1–7, each test-driven), then the long runs in four queued batches (Tasks 8–11), then the write-up (Task 12). New datasets, checkpoints and run outputs live on the internal disk under `~/capstone_data/` and `~/capstone_runs/hardening/`; only JSON summaries and markdown are copied into the repo.

**Tech Stack:** Python 3, PyTorch, Opacus RDP accountant, Flower 1.x simulation (Ray) with SecAgg+, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-hardening-design.md` (approved 2026-11-17). Parent spec: `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md`.

## Global Constraints

- One machine, 8 cores, 8 GB RAM: one training job at a time; long runs detached with `perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' /bin/sh -c '…' > LOG 2>&1 < /dev/null &` plus `caffeinate -ims -w <sh pid>`.
- New datasets and checkpoints go to `~/capstone_data/` and `~/capstone_runs/hardening/` (internal disk: the USB drive has ~7 GB free and remounts kill open files). HI-Small and LI-Small (1 client) stay where they are under `data/`.
- Never commit raw data, `*.pt` or `*.npz`. Push only to remote `fork`, never `origin`.
- Long runs are never restarted from scratch; rerun only what is missing.
- Features stay fixed transforms; δ = 1e-6; account-level DP; RDP accounting; 10 clients; chronological 60/20/20.
- Seeded DP randomness only with `--seeded-dp` (tests); every new run JSON records `dp_randomness`.
- `results/hardening/tuned_settings.json` is committed before any HI-Small fine-tuning run of this stage.
- Delete `._*` files before git operations: `find . -name '._*' -not -path './data/*' -delete`.

## Review Focus

1. A SAML-D currency or payment string spelled differently from the mapping silently becomes "unknown". Expected: the builder records per-column unknown counts in the metadata, and the SAML-D test pins every value present in the real file. Test in Task 3.
2. A production run that silently uses seeded DP randomness. Expected: secure is the default of `parse_args`, and the JSON says which was used. Test in Task 1.
3. `--settings` with an ε that the tuned file does not contain. Expected: a `SystemExit` naming the missing level before any training, never a fallback to command-line defaults. Test in Task 4.
4. Flower head-DP mode inheriting the 20,000-account cap or dropping fraud-rows-first, and so silently differing from the in-process run. Expected: no cap, fraud rows first. Test in Task 6.
5. An account-hash pseudo-bank with no laundering in its train split, which would crash the per-client metrics. Expected: build-time stats show it, and tuning uses pooled validation PR-AUC. Test in Task 3.

---

### Task 1: Secure DP randomness

**Files:**
- Create: `src/federated/dp_random.py`
- Modify: `src/federated/dp_sgd.py` (`_sample_rows`, `train_local_account_dp`), `experiments/run_ibm_comparison.py` (`parse_args`, `local_fit`, `run_federated` privacy dict)
- Test: `tests/test_dp_random.py`; add `"--seeded-dp"` to existing test arg lists that pin exact DP numbers (`tests/test_run_ibm_comparison.py`, `tests/test_run_dp_finetune.py`, `tests/test_flower_app.py`, `tests/test_run_flower.py`, `tests/test_dp_sgd.py` as needed)

**Interfaces:**
- Produces: `class SeededDPRandom(generator: torch.Generator)` and `class SecureDPRandom()`, both with `uniform(n: int) -> torch.Tensor` (float64 in [0, 1)), `permutation(n: int) -> torch.Tensor` (int64), `gaussian(std: float, shape) -> torch.Tensor` (float32). `make_dp_random(args, generator) -> SeededDPRandom | SecureDPRandom`, which picks seeded iff `args.seeded_dp`. `train_local_account_dp(..., generator=None, dp_rng=None)`: when `dp_rng` is None it uses `SeededDPRandom(generator)`, so existing callers and golden tests are unchanged. `parse_args` gains `--seeded-dp` (store_true, default False). The `run_federated` privacy dict gains `"dp_randomness": "secure" | "seeded"` per client.
- Seeded must reproduce today's draws exactly: `uniform` = `torch.rand(n, generator=g)`, `permutation` = `torch.randperm(n, generator=g)`, `gaussian` = `torch.normal(0.0, std, shape, generator=g)`.
- Secure: uniform from `os.urandom(8 * n)` as uint64 `>> 11` times `2**-53`; permutation = `argsort` of a secure uniform; gaussian = Box–Muller in float64 from secure uniforms (guard `u1 > 0`), then Opacus's floating-point guard: the sum of 4 standard normals divided by 2, times `std`, cast to float32.

- [ ] **Step 1: Write the failing tests** in `tests/test_dp_random.py`:
  - `test_seeded_matches_torch_draws`: with `torch.Generator().manual_seed(3)`, every `SeededDPRandom` method equals the matching torch call on a twin generator.
  - `test_secure_gaussian_moments`: 200,000 draws at std 2.0: `abs(mean) < 0.02`, `abs(std - 2.0) < 0.02`, `scipy.stats.kstest(x / 2.0, "norm").pvalue > 1e-3`.
  - `test_secure_uniform_range_and_permutation`: uniform in [0, 1); `sorted(permutation(50)) == list(range(50))`.
  - `test_secure_draws_differ_across_calls`: two `gaussian(1.0, (100,))` calls are not equal.
  - `test_parse_args_defaults_to_secure`: `parse_args([]).seeded_dp is False` and `make_dp_random(args, g)` is a `SecureDPRandom`.
  - `test_account_dp_run_records_randomness`: run `run_federated` on `tiny3` with `account_dp_eps=8` (rounds 1, as `tests/test_run_ibm_comparison.py` does), once without and once with `--seeded-dp`: every client's privacy dict has `dp_randomness` "secure", then "seeded".
- [ ] **Step 2: Run** `python3 -m pytest tests/test_dp_random.py -q -p no:warnings`. Expected: FAIL with ImportError on `src.federated.dp_random`.
- [ ] **Step 3: Implement `dp_random.py`**, route `_sample_rows` (`torch.rand` → `uniform`, `torch.randperm` → `permutation`) and the noise (`torch.normal` → `gaussian`) through `dp_rng`, and have `local_fit` pass `make_dp_random(args, generator)`. Client-level DP (A0's `client_dp`) is not touched; Stage H does not run it.
- [ ] **Step 4: Add `"--seeded-dp"`** to every existing test invocation that runs account DP and compares exact numbers. Run the whole suite: `python3 -m pytest -q -p no:warnings`. Expected: all pass (85 before this task, plus the new ones).
- [ ] **Step 5: Commit**

```bash
git add src/federated/dp_random.py src/federated/dp_sgd.py experiments/run_ibm_comparison.py tests/
git commit -m "Secure DP randomness by default: noise, account sampling and row choice from os.urandom"
```

### Task 2: Alert-budget metrics

**Files:**
- Modify: `src/utils/metrics.py` (`fraud_metrics`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `fraud_metrics` adds `precision_at_1000`, `recall_at_1000`, `precision_at_5000` and `recall_at_5000`. Alerts are the top-K scores by `np.argsort(-y_prob, kind="stable")`, with K capped at n. Recall is `None` when there is no positive. `mean_std` picks the new keys up automatically.

- [ ] **Step 1: Write the failing test** `test_alert_budget_metrics`: y = 6,000 rows with 10 positives placed at the 10 highest scores, plus 5 positives at the lowest. Assert `precision_at_1000 == 10 / 1000`, `recall_at_1000 == 10 / 15`, `precision_at_5000 == 10 / 5000`, `recall_at_5000 == 10 / 15`. A second case with n = 300 asserts K is capped (`precision_at_1000 == positives / 300`).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_metrics.py -q -p no:warnings`. Expected: FAIL with KeyError `precision_at_1000`.
- [ ] **Step 3: Implement** in `fraud_metrics`.
- [ ] **Step 4: Run** the whole suite. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Metrics: precision and recall at 1,000 and 5,000 alerts"`.

### Task 3: SAML-D and AMLSim builders, account-hash clients

**Files:**
- Create: `src/data/public_sources.py`
- Modify: `src/data/ibm_aml.py` (`build_dataset`), `experiments/build_ibm_aml.py` (flags)
- Test: `tests/test_public_sources.py`

**Interfaces:**
- Produces: `load_saml_d(csv_path: Path) -> pd.DataFrame` and `load_amlsim(dir_path: Path) -> pd.DataFrame`. Both return exactly the columns and dtypes of `load_transactions`, time-sorted (stable), plus `df.attrs["unknown"] = {"pay_currency": int, "recv_currency": int, "pay_format": int}`.
- `build_dataset(csv_path, out_dir, n_clients=10, k=K_HISTORY, source="ibm", group_by="bank")`, where `source` ∈ {"ibm", "saml_d", "amlsim"} and `group_by` ∈ {"bank", "account_hash"}. `account_hash` sets the client to `from_acct % n_clients`. Metadata gains `"source_format": source`, `"group_by": group_by` and `"unknown_categories"` (from `attrs`; zeros for IBM). With `account_hash`, `n_banks` and `bank_to_client` hold the account-hash description: `n_banks` = number of distinct sending accounts, `bank_to_client` = `{}`.
- CLI: `build_ibm_aml.py --source {ibm,saml_d,amlsim} --group-by {bank,account_hash}`.
- SAML-D mapping (spec table): ts = `Date` + `Time` as UTC seconds. Banks = sender/receiver bank location strings, factorized over both columns. Accounts = `Sender_account` / `Receiver_account` factorized together. Both amounts = `Amount`. Label = `Is_laundering`. Currency map: `{"UK pounds": "UK Pound", "Euro": "Euro", "Swiss franc": "Swiss Franc", "Yen": "Yen", "US dollar": "US Dollar", "Mexican Peso": "Mexican Peso", "Indian rupee": "Rupee", "Pakistani rupee": "Rupee"}`; anything else → -1. Format map: `{"Credit card": "Credit Card", "Debit card": "Credit Card", "Cheque": "Cheque", "ACH": "ACH", "Cross-border": "Wire", "Cash Deposit": "Cash", "Cash Withdrawal": "Cash"}`.
- AMLSim mapping: ts = `1640995200 + TIMESTAMP * 86400`, one bank (code 0), accounts from `SENDER_ACCOUNT_ID` / `RECEIVER_ACCOUNT_ID`, both amounts = `TX_AMOUNT`, currency "US Dollar", format "ACH", label = `IS_FRAUD` ("True"/"False" or bool).

- [ ] **Step 1: Write the failing tests**:
  - `test_saml_d_mapping`: a 6-row CSV written in the test, covering UK pounds/Dirham, Cross-border/Cash Deposit, a UK→UAE row and two rows of one sender. Assert the ts of `2022-10-07 10:35:19` is `1665138919`; cross-location rows have `from_bank != to_bank`; `pay_currency` is the `CURRENCIES.index("UK Pound")` code; Dirham is -1; `attrs["unknown"]["recv_currency"] == 1`; Cross-border is the Wire code; Cash Deposit is the Cash code; both amounts are equal; the label is carried.
  - `test_saml_d_real_values_all_mapped_or_counted`: skipped unless `/Volumes/Untitled/datasets_capstone/SAML-D.csv` exists. Reads only `Payment_currency`, `Received_currency` and `Payment_type` with `usecols` and asserts every distinct `Payment_type` maps (no unknown formats). It prints the currency names that fall to unknown, for the gate file.
  - `test_amlsim_mapping`: a 3-row `transactions.csv` written in the test. Assert ts step 2 → `1640995200 + 172800`, one bank, ACH and US Dollar codes, and boolean labels.
  - `test_account_hash_grouping`: `build_dataset(..., source="saml_d", group_by="account_hash", n_clients=3)` on a 60-row synthetic SAML-D CSV. Every sending account's rows land in exactly one client; `metadata["group_by"] == "account_hash"`; per-client stats exist even for a client with no fraud in train (Review Focus 5).
  - `test_ibm_build_unchanged`: an IBM build of the `tiny_csv` fixture produces the same `table.pt` tensors as before (compare against a build with `source="ibm", group_by="bank"` passed explicitly, and check `metadata["unknown_categories"]` is all zeros).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_public_sources.py -q -p no:warnings`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** the loaders and the `build_dataset` dispatch. The loader returns the frame; `encode_features` and `time_split` are reused unchanged.
- [ ] **Step 4: Run** the new tests, then the whole suite. Expected: all pass. Paste the printed unknown-currency list into the ledger.
- [ ] **Step 5: Commit** `git commit -m "Builders for SAML-D and AMLSim in the IBM format; account-hash clients"`.

### Task 4: Fixed-rounds fine-tuning with tuned settings, saved scores

**Files:**
- Modify: `experiments/run_ibm_comparison.py` (`run_federated`), `experiments/run_dp_finetune.py`
- Test: `tests/test_run_dp_finetune.py`

**Interfaces:**
- Consumes: Task 1 `make_dp_random`; Task 2 metrics.
- Produces: `run_federated(..., include_initial=False, select="val", return_probs=False)`. With `select="last"` the reported metrics are those of the model after round `args.rounds`. In that case `metrics["selected_round"] = args.rounds`, and `metrics["val_selected"] = {"round": r, "pooled": <pooled test metrics of the val-best state>}` is reported for comparison only. With `return_probs=True`, `metrics["test_probs"] = {client: np.ndarray}` for the reported model.
- `embed_clients(backbone, clients) -> dict` (moved out of `main`, same output as today's `emb`). `run_dp_finetune.py` keeps a module-level `HeadModel`, `load_pretrained`, `embed` and `EmbeddingDataset` (Tasks 5–6 import them).
- New `run_dp_finetune.py` flags:
  - `--settings PATH`: a tuned-settings JSON (Task 5 format). For `head_dp` at ε=e it sets `args.lr`, `args.rounds` and the trained layers from `chosen[str(float(e))]`; for `head_nodp` it uses `chosen["nodp"]`. A missing key raises `SystemExit("tuned settings have no entry for <key>")` before any training (Review Focus 3). With `--settings`, selection is `"last"`, and the JSON records `"settings": <file contents>`, `"selection": "fixed_rounds"`.
  - `--save-scores DIR`: writes `DIR/<run name>_seed<seed>.npz` holding one array per client (`client_<c>`), then drops `test_probs` from the JSON.
- `full_nodp`, `zero_shot` and runs without `--settings` keep today's behaviour (`selection: "val"`).

- [ ] **Step 1: Write the failing tests**:
  - `test_fixed_rounds_reports_last_round(tiny3)`: with a monkeypatched `local_fit` that makes each round worse (as in `test_initial_model_can_be_selected`), `select="last"` with rounds 2 gives `selected_round == 2`, `val_selected["round"] == 0` and a lower pooled PR-AUC than the val-selected one.
  - `test_settings_applied_per_level(ckpt, tiny3, tmp_path)`: a settings file with ε 8 → `{"lr": 0.003, "layers": "classifier", "rounds": 2}` and nodp → `{"lr": 0.001, "layers": "last_layer", "rounds": 1}`. Assert the `head_dp_eps8.0` privacy steps equal `2 * steps_per_round`, `trained_parameters["head_dp_eps8.0"] == 1089`, `trained_parameters["head_nodp"] == 33` and `out["selection"] == "fixed_rounds"`.
  - `test_settings_missing_level_exits(ckpt, tiny3, tmp_path)`: `--epsilons 3` against the file above raises `SystemExit` whose message contains `3.0`.
  - `test_save_scores(ckpt, tiny3, tmp_path)`: `--save-scores` writes `head_dp_eps8.0_seed0.npz` with one array per client, each the length of that client's test split.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_run_dp_finetune.py -q -p no:warnings`. Expected: the 4 new tests FAIL (unknown argument / missing key).
- [ ] **Step 3: Implement.** Because `trained_parameters` must differ per level, key it by run name (`head_dp_eps8.0`, `head_nodp`) when `--settings` is given.
- [ ] **Step 4: Run** the whole suite. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Fine-tuning: fixed rounds from tuned public settings, saved test scores"`.

### Task 5: Public tuning script

**Files:**
- Create: `experiments/tune_public.py`
- Test: `tests/test_tune_public.py`

**Interfaces:**
- Consumes: Task 4 `embed_clients`, `HeadModel`, `load_pretrained`, `run_federated(select="last")`.
- Produces: `choose(rows: list[dict]) -> dict`. Each row is `{"level", "lr", "layers", "rounds", "val_pr_auc"}`; the result maps each level to the best row by `val_pr_auc`. Ties break toward fewer rounds, then `"last_layer"`, then the smaller lr.
- CLI: `python3 experiments/tune_public.py --ckpt C --data-path TUNE_DATA --out results/hardening/tune_<name>.json [--epsilons 1 3 8] [--lrs 3e-4 1e-3 3e-3] [--layers last_layer classifier] [--rounds-grid 1 2 5 10] [any run_ibm_comparison flag]`. It embeds once and runs no-DP plus each ε over the full grid with `select="last"`, recording the tuning set's pooled validation PR-AUC at the end of each run (its test split is never scored). Output JSON: `{"backbone": C, "tuned_on": TUNE_DATA, "grid": rows, "chosen": {"1.0": {...}, "3.0": {...}, "8.0": {...}, "nodp": {...}}, "dp_randomness": ...}`. `chosen` is exactly the format `run_dp_finetune.py --settings` reads.

- [ ] **Step 1: Write the failing tests**:
  - `test_choose_tie_break`: rows that tie on `val_pr_auc` pick rounds 1 over 5, then `last_layer` over `classifier`, then lr 3e-4 over 1e-3; a strictly better row wins regardless.
  - `test_tune_public_end_to_end(ckpt, tiny3, tmp_path)`: grid `--epsilons 8 --lrs 1e-3 --layers last_layer --rounds-grid 1 2` with `--seeded-dp` and the small flags. The JSON has 4 grid rows (2 levels × 2 rounds); `chosen` has keys `{"8.0", "nodp"}`; feeding that file to `run_dp_finetune.main([... "--settings", path, "--epsilons", "8"])` runs without error.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_tune_public.py -q -p no:warnings`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the whole suite. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Tune fine-tuning settings on public data only"`.

### Task 6: Flower head-DP mode

**Files:**
- Modify: `src/federated/flower_app.py`, `experiments/run_flower.py`
- Test: `tests/test_flower_app.py`, `tests/test_run_flower.py`; move the `ckpt` fixture from `tests/test_run_dp_finetune.py` to `tests/conftest.py`

**Interfaces:**
- Consumes: Task 4 `HeadModel`, `load_pretrained`, `embed_clients`; Task 1 secure randomness (via `local_fit`).
- Produces: `FlowerSettings` gains `ckpt: str | None = None` and `head_train: str = "last_layer"`. When `ckpt` is set:
  - `prepare_head_embeddings(settings) -> None` (called by `run` before the simulation) writes `out_dir/emb/client_<c>.pt` with the train/val/test embeddings, labels, accounts and `n_train_accounts`. This is a simulation shortcut: in deployment each bank embeds its own data with the public backbone.
  - `client_data` loads those files, with no account cap.
  - `build_model_for` returns a `HeadModel` builder (prior bias from the metadata's pooled train rate, `set_trainable(head_train)`).
  - `BankClient.fit` uses `AccountDPConfig(..., fraud_rows_first=True)`.
- `run_flower.py --ckpt C [--head-train last_layer|classifier] [--settings tuned.json]`. With `--settings` and `--account-dp-eps e`, lr, rounds and head layers come from `chosen[str(e)]`, and a missing key exits as in Task 4. In head mode the default `--max-train-accounts 20000` is not added. The result JSON gains `"mode": "head_dp"`, `"ckpt"`, `"head_train"` and `"dp_randomness"`; its metrics use the final round (`selection: "fixed_rounds"`) when `--settings` is given.

- [ ] **Step 1: Write the failing tests**:
  - `test_head_mode_trains_only_head(ckpt, tiny3)`: `BankClient(FlowerSettings(..., ckpt=ckpt, account_dp_eps=8, run_args=[... "--seeded-dp"])).fit(params, {"server_round": 1})` returns 2 arrays (the last layer's weight and bias) totalling 33 values.
  - `test_head_mode_no_cap_and_fraud_rows_first(ckpt, tiny3, monkeypatch)`: `client_data` reports the uncapped `n_train_accounts`, and the captured `AccountDPConfig.fraud_rows_first is True` (Review Focus 4).
  - `test_run_flower_head_dp_end_to_end(ckpt, tiny3, tmp_path)`: rounds 2, SecAgg+ on, `--debug-dump-locals`, ε 8, `--seeded-dp`. Every `aggregation_errors` value is ≤ `error_bound`, `out["mode"] == "head_dp"`, and every client's `privacy[c]["epsilon_spent"] <= 8.0 + 1e-6`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_flower_app.py tests/test_run_flower.py -q -p no:warnings`. Expected: the new tests FAIL (unexpected keyword `ckpt`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the whole suite. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Flower + SecAgg+ head-DP mode on a pretrained backbone"`.

### Task 7: Client 0 analysis

**Files:**
- Create: `experiments/analyze_client0.py`
- Test: `tests/test_analyze_client0.py`

**Interfaces:**
- Consumes: `load_transactions`, `assign_banks_to_clients`, `time_split` (ibm_aml); scores saved by Task 4 (`client_0` array, test-split order).
- Produces: `per_bank_report(df_client: pd.DataFrame, split: np.ndarray, test_probs: np.ndarray) -> list[dict]`. `df_client` holds one client's rows in builder order; the function returns, per sending bank, `{"bank", "train_n", "train_fraud", "test_n", "test_fraud", "share_of_client", "test_pr_auc" (None if a class is absent)}`, sorted by `test_n` descending. CLI: `analyze_client0.py --raw HI-Small_Trans.csv --scores DIR/head_dp_eps8.0_seed0.npz --client 0 --out results/hardening/client0.json`. It rebuilds the client assignment exactly as `build_dataset` does and asserts the row count equals the saved scores' length.

- [ ] **Step 1: Write the failing tests**:
  - `test_per_bank_report(tiny_csv)`: on the IBM tiny fixture, banks' `test_n` sum to the client's test rows, `share_of_client` sums to 1, and a bank whose test split has no fraud gets `test_pr_auc is None`.
  - `test_alignment_check`: scores of the wrong length raise `ValueError`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_analyze_client0.py -q -p no:warnings`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the whole suite. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Client 0 per-bank analysis"`.

### Task 8: Builds and pretraining (batch 1, ~12 h)

**Files:** none in the repo. Outputs in `~/capstone_data/` and `~/capstone_runs/hardening/`. Log: `~/capstone_runs/hardening/batch1.log`.

- [ ] **Step 1: Confirm nothing else is training** (`pgrep -fl "run_ibm_comparison|run_dp_finetune|pretrain_public|run_flower"`; expected: no output). Confirm free space: `df -h ~` shows > 20 GB.
- [ ] **Step 2: Launch batch 1 detached**, one chain, in this order:

```bash
D=~/capstone_data; R=~/capstone_runs/hardening; S=/Volumes/Untitled/datasets_capstone
python3 experiments/build_ibm_aml.py --source saml_d --src $S/SAML-D.csv --out $D/saml_d --clients 1
python3 experiments/build_ibm_aml.py --source saml_d --group-by account_hash --src $S/SAML-D.csv --out $D/saml_d_10 --clients 10
python3 experiments/build_ibm_aml.py --src $S/ibm_aml/li_small/LI-Small_Trans.csv --out $D/ibm_li_small_10 --clients 10
python3 experiments/build_ibm_aml.py --source amlsim --src $S/ibmamlsim --out $D/amlsim --clients 1
python3 -W ignore experiments/pretrain_public.py --data-path $D/saml_d --out-ckpt $R/pretrain_saml_d.pt --out-json $R/pretrain_saml_d.json
python3 -W ignore experiments/pretrain_public.py --data-path $D/amlsim --out-ckpt $R/pretrain_amlsim.pt --out-json $R/pretrain_amlsim.json
python3 -W ignore experiments/pretrain_public.py --max-train-accounts 67553 --out-ckpt $R/pretrain_li_10pct.pt --out-json $R/pretrain_li_10pct.json
python3 -W ignore experiments/pretrain_public.py --max-train-accounts 6755 --out-ckpt $R/pretrain_li_1pct.pt --out-json $R/pretrain_li_1pct.json
python3 -W ignore experiments/pretrain_public.py --seed 1 --out-ckpt $R/pretrain_li_seed1.pt --out-json $R/pretrain_li_seed1.json
python3 -W ignore experiments/pretrain_public.py --seed 2 --out-ckpt $R/pretrain_li_seed2.pt --out-json $R/pretrain_li_seed2.json
```

  The LI-Small source path must be the one `data/ibm_li_small/metadata/dataset_info.json` records as `source`; check it before launching. 67,553 and 6,755 are 10% and 1% of LI-Small's 675,527 train accounts.
- [ ] **Step 3: When the chain ends**, check that every JSON exists and that each pretraining log has 5 epoch lines. Record each backbone's validation and test PR-AUC in the ledger. Copy `pretrain_*.json` to `results/hardening/` (`.gitignore` gains `!results/hardening/*.json` in this step). Expected: SAML-D pretraining test PR-AUC well above its test laundering rate; the 1% backbone may be near chance (spec: a result, not a failure).
- [ ] **Step 4: Commit** `git commit -m "Stage H batch 1: public builds and pretraining results"`.

### Task 9: Cross-tuning (batch 2, ~8 h)

Tasks 9–12 use the same `D=~/capstone_data` and `R=~/capstone_runs/hardening` as Task 8, and run from the repo root.

- [ ] **Step 1: Launch detached**, logging to `~/capstone_runs/hardening/batch2.log`:

```bash
python3 -W ignore experiments/tune_public.py --ckpt checkpoints/pretrain_li_small.pt --data-path $D/saml_d_10 --out $R/tune_li_small.json
python3 -W ignore experiments/tune_public.py --ckpt $R/pretrain_saml_d.pt --data-path $D/ibm_li_small_10 --out $R/tune_saml_d.json
```

- [ ] **Step 2: Check** that each JSON has 4 × 2 × 3 × 4 = 96 grid rows and `chosen` has keys `1.0, 3.0, 8.0, nodp`. If any chosen setting sits at the edge of the grid (lr 3e-3 or 3e-4, or 10 rounds), note it in the ledger; do not extend the grid (spec: fixed before the runs).
- [ ] **Step 3: Write `results/hardening/tuned_settings.json`** as `{"li_small": <tune_li_small chosen>, "saml_d": <tune_saml_d chosen>}`, plus the two tuning JSONs. Commit **before** Task 10: `git commit -m "Stage H: fine-tuning settings tuned on public data only"`, then `git push fork stage-a0-core-pipeline`, so the commit is timestamped on the remote before any HI-Small run.

### Task 10: HI-Small fine-tuning and Flower (batch 3, ~12 h)

- [ ] **Step 1: Extract per-backbone settings files** `$R/settings_li_small.json` and `$R/settings_saml_d.json` (each `{"chosen": …}` from the tuning JSON).
- [ ] **Step 2: Launch detached**, logging to `batch3.log`. Every command uses `--settings`; the first adds `--save-scores $R/scores`:

```bash
FT="python3 -W ignore experiments/run_dp_finetune.py --epsilons 1 3 8"
$FT --ckpt checkpoints/pretrain_li_small.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --seeds 0 1 2 --save-scores $R/scores --out $R/ft_li_small.json
$FT --ckpt $R/pretrain_saml_d.pt --settings $R/settings_saml_d.json --models zero_shot head_nodp head_dp --seeds 0 1 2 --out $R/ft_saml_d.json
$FT --ckpt $R/pretrain_saml_d.pt --models full_nodp --seeds 0 --out $R/ft_saml_d_full.json
$FT --ckpt $R/pretrain_li_10pct.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --seeds 0 1 2 --out $R/ft_li_10pct.json
$FT --ckpt $R/pretrain_li_10pct.pt --models full_nodp --seeds 0 --out $R/ft_li_10pct_full.json
$FT --ckpt $R/pretrain_li_1pct.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --seeds 0 1 2 --out $R/ft_li_1pct.json
$FT --ckpt $R/pretrain_li_1pct.pt --models full_nodp --seeds 0 --out $R/ft_li_1pct_full.json
$FT --ckpt $R/pretrain_li_seed1.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --seeds 0 --out $R/ft_li_seed1.json
$FT --ckpt $R/pretrain_li_seed2.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --seeds 0 --out $R/ft_li_seed2.json
python3 -W ignore experiments/run_dp_finetune.py --ckpt $R/pretrain_amlsim.pt --settings $R/settings_li_small.json --models zero_shot head_nodp head_dp --epsilons 8 --seeds 0 --out $R/ft_amlsim.json
python3 -W ignore experiments/run_flower.py --ckpt checkpoints/pretrain_li_small.pt --settings $R/settings_li_small.json --account-dp-eps 8 --debug-dump-locals --work-dir $R/flower_head --out $R/flower_head_dp_eps8.json
```

- [ ] **Step 3: Check:**
  - every JSON's `selection` and `dp_randomness` fields;
  - every `epsilon_spent` ≤ its target;
  - Flower: every aggregation error ≤ bound, and pooled test PR-AUC within 0.005 of `ft_li_small.json`'s `head_dp_eps8.0` seed 0 (spec gate). If the PR-AUC check fails, use systematic-debugging before writing anything.

  Delete `$R/flower_head/locals` and `$R/flower_head/emb`. Copy the JSONs to `results/hardening/`.
- [ ] **Step 4: Commit** `git commit -m "Stage H batch 3: HI-Small fine-tuning under public-only settings, Flower head DP"`.

### Task 11: A0 error bars and FedProx μ (batch 4, ~12.5 h)

- [ ] **Step 1: Launch detached**, logging to `batch4.log`:

```bash
A0="python3 -W ignore experiments/run_ibm_comparison.py --max-train-accounts 20000"
$A0 --models centralized gbdt fedavg --seeds 1 2 --out $R/a0_seeds12.json
$A0 --models fedprox --mu 0.001 --seeds 0 --out $R/a0_fedprox_mu0.001.json
$A0 --models fedprox --mu 0.01 --seeds 0 --out $R/a0_fedprox_mu0.01.json
```

- [ ] **Step 2: Check** that the pooled metrics and train times are present. Merge seed 0 from `results/comparison/ibm_hi_small.json` with seeds 1–2 to get mean ± std for `centralized`, `gbdt` and `fedavg` (written into `results/hardening/a0_three_seeds.json`, with the source of each seed recorded). Copy the μ JSONs.
- [ ] **Step 3: Commit** `git commit -m "Stage H batch 4: A0 three-seed baselines, FedProx mu sweep"`.

### Task 12: Analysis, write-up, push

**Files:**
- Create: `results/hardening/HARDENING_GATE.md`, `results/hardening/client0.json`
- Modify: `README.md`, `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md` (Stage H row in the stage table), `.gitignore` (already done in Task 8)

- [ ] **Step 1: Client 0:**

```bash
python3 experiments/analyze_client0.py --raw /Volumes/Untitled/datasets_capstone/ibm_aml/hi_small/HI-Small_Trans.csv --scores $R/scores/head_dp_eps8.0_seed0.npz --client 0 --out results/hardening/client0.json
```

- [ ] **Step 2: Literature check.** Find the published HI-Small results for Multi-GNN (Egressy et al.) and for boosting with graph features. Record for each: metric, split, number and citation. If their metric is minority-class F1 on a split we can reproduce from our data, compute ours the same way from saved scores. Otherwise state why the numbers are not comparable.
- [ ] **Step 3: Write `HARDENING_GATE.md`** answering the spec's Q1–Q6 with the pre-registered verdicts, which are fixed and not adjusted:
  - **DP keeps the utility:** head_dp_eps8 mean ≥ 95% of the best no-DP PR-AUC on the same backbone.
  - **The result is useful:** head_dp_eps8 > A0 `centralized` three-seed mean.

  It also covers alert budgets at 1,000 and 5,000 alerts, client 0, the literature table, and every row's `dp_randomness` and `selection`.
- [ ] **Step 4: Update the README** headline per the spec's narrowing rules, and add the Stage H row to the v2 spec.
- [ ] **Step 5: Run the whole suite** (`python3 -m pytest -q -p no:warnings`; expected: all pass), commit `git commit -m "Stage H gate: hardened results"`, then `git push fork stage-a0-core-pipeline`.
