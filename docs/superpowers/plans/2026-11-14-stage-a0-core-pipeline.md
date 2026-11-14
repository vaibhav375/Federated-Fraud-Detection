# Stage A0: Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the existing Temporal Graph Transformer on IBM AML HI-Small with honest chronological splits, account-level differential privacy, and the baselines needed to show a federation gain.

**Architecture:** A new data builder turns the raw IBM CSV into one indexed table per bank group (per-transaction features plus a causal index of each account's earlier transactions), so sequences are gathered at batch time and fit in 8 GB of RAM. A new DP module computes per-example gradients with `torch.func` on the unmodified model, clips per account, adds Gaussian noise and tracks ε per bank with Opacus's Rényi accountant. One new experiment script runs every baseline and DP setting on the same splits and writes one JSON file.

**Tech Stack:** Python 3.12, PyTorch 2.8 (`torch.func`), Opacus 1.5.4 (accountant only), pandas, scikit-learn (`HistGradientBoostingClassifier`), pytest 9.

**Spec:** `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md`

## Global Constraints

- Runs on an Apple M1 with 8 GB RAM, CPU only. No step may hold all sequences as a dense tensor; peak memory of the data build stays under 6 GB.
- No new dependencies. Add `pytest>=8.0` to `requirements.txt` (already installed).
- No SMOTE or any synthetic rows. No statistic fitted on the data enters a feature (no scaler, no learned vocabulary).
- Splits are chronological 60/20/20 with global time cut-offs and are used exactly as built. Nothing re-splits them.
- Privacy unit is the sending account. δ = 1e-6, and a run aborts if δ ≥ 1 / (train accounts of any client).
- Model selection uses validation data only. Test data is read once per run, for the selected round.
- Headline metrics are PR-AUC and recall at 1% false-positive rate.
- Existing files `run_training.py`, `experiments/run_model_comparison.py`, `experiments/run_epsilon_sweep.py`, `backend/`, `frontend/` and every `src/models/*.py` are not modified. No state-dict key changes.
- Raw CSVs live under `datasets_capstone/ibm_aml/` (already git-ignored) and are never committed.
- All commands run from the repo root. Tests run with `python3 -m pytest tests -q`.
- Work on branch `stage-a0-core-pipeline`.

## Review Focus

1. **A category value not in the hard-coded vocabulary** (new currency or payment format): the row gets an all-zero one-hot and the build continues. Test in Task 3.
2. **A client or split with no fraud, or with one class only:** metrics return `None` for PR-AUC and ROC-AUC, never raise. Test in Task 4.
3. **Timestamp ties** at a split boundary or inside one account: tied rows land in the same split; history order follows original row order. Tests in Tasks 2 and 3.
4. **An empty Poisson batch under DP:** the step still adds noise and still advances the accountant. Test in Task 5.
5. **A test period whose fraud rate differs sharply from the training period** (a known risk with simulated data that ends in a laundering-heavy tail): the build records per-split fraud rates and prints a warning when any split's rate is more than 5× another's. Test in Task 3.

## File Structure

| File | Responsibility |
|---|---|
| `src/data/ibm_aml.py` (create) | Raw IBM schema, loading, bank-to-client assignment, time split, feature encoding, dataset build |
| `src/data/sequences.py` (create) | Dataset-agnostic history index and lazy sequence dataset; loader for built clients |
| `src/utils/metrics.py` (create) | Threshold-free and threshold-tuned fraud metrics |
| `src/federated/dp_sgd.py` (create) | Account-level DP-SGD: per-example gradients, per-account clipping, noise, accounting, local training loop |
| `experiments/build_ibm_aml.py` (create) | CLI wrapper around the build and the raw-file profile |
| `experiments/run_ibm_comparison.py` (create) | Every baseline and DP setting on the built dataset; writes the results JSON |
| `tests/conftest.py`, `tests/test_*.py` (create) | Tests |

---

### Task 1: Raw data loading and schema check

**Prerequisite (person, not agent):** download `HI-Small_Trans.csv` from the Kaggle dataset "IBM Transactions for Anti Money Laundering (AML)" into `datasets_capstone/ibm_aml/`. The slug is believed to be `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`. Tasks 1–6 can be built and tested without the file; Task 7 cannot.

**Files:**
- Create: `src/data/ibm_aml.py`, `tests/conftest.py`, `tests/test_ibm_aml_load.py`, `tests/test_ibm_aml_real.py`
- Modify: `requirements.txt` (add `pytest>=8.0`)

**Interfaces:**
- Produces:
  - `RAW_COLUMNS: list[str]` = `["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]`
  - `PAYMENT_FORMATS: list[str]` = `["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]`
  - `CURRENCIES: list[str]` = `["US Dollar", "Euro", "Yuan", "Yen", "UK Pound", "Rupee", "Ruble", "Canadian Dollar", "Australian Dollar", "Swiss Franc", "Brazil Real", "Mexican Peso", "Saudi Riyal", "Shekel", "Bitcoin"]`
  - `load_transactions(csv_path: Path) -> pd.DataFrame` with columns `ts` (int64 seconds), `from_bank`, `to_bank` (int32), `from_acct`, `to_acct` (int32 codes; an account is the pair bank + account string, one code space for both sides), `amount_paid`, `amount_received` (float32), `pay_currency`, `recv_currency`, `pay_format` (int8 index into the lists above, `-1` if absent), `label` (int8). Rows sorted by `ts` with a stable sort, index reset.
  - Test helpers in `tests/helpers.py`: `make_raw(n_rows=400, n_banks=6, accounts_per_bank=5, n_fraud=12, seed=0) -> pd.DataFrame` in `RAW_COLUMNS`, timestamps one minute apart from `2022/09/01 00:00` in format `%Y/%m/%d %H:%M`, the sender cycling through every bank/account pair in order (so each account appears in every split), fraud rows evenly spaced in time; `write_raw(df, tmp_path) -> Path`, which writes the CSV with the duplicate `Account` header the real file has.
  - Fixture in `tests/conftest.py`: `tiny_csv(tmp_path) -> Path` = `write_raw(make_raw(), tmp_path)`.

The three constant lists and the timestamp format are from memory. The real-file test below is what confirms them; if it fails, correct the constants to match the file and nothing else.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ibm_aml_load.py
def test_load_columns_and_dtypes(tiny_csv):
    df = load_transactions(tiny_csv)
    assert list(df.columns) == ["ts", "from_bank", "to_bank", "from_acct", "to_acct",
                                "amount_paid", "amount_received", "pay_currency",
                                "recv_currency", "pay_format", "label"]
    assert len(df) == 400 and int(df["label"].sum()) == 12
    assert df["ts"].is_monotonic_increasing
    assert df["amount_paid"].dtype == np.float32

def test_same_account_string_at_two_banks_gets_two_codes(tmp_path):
    raw = make_raw(n_rows=4, n_banks=2, accounts_per_bank=1, n_fraud=0)
    raw["Account"] = "A1"; raw["From Bank"] = [1, 2, 1, 2]
    df = load_transactions(write_raw(raw, tmp_path))
    assert df["from_acct"].nunique() == 2

def test_unknown_category_is_minus_one(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0); raw.loc[0, "Payment Format"] = "Carrier Pigeon"
    assert load_transactions(write_raw(raw, tmp_path))["pay_format"].iloc[0] == -1

def test_tied_timestamps_keep_file_order(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0)
    raw["Timestamp"] = "2022/09/01 00:00"; raw["Amount Paid"] = [1.0, 2.0, 3.0, 4.0]
    assert load_transactions(write_raw(raw, tmp_path))["amount_paid"].tolist() == [1.0, 2.0, 3.0, 4.0]

# tests/test_ibm_aml_real.py
RAW = Path("datasets_capstone/ibm_aml/HI-Small_Trans.csv")

@pytest.mark.skipif(not RAW.exists(), reason="raw IBM file not downloaded")
def test_real_file_matches_assumed_schema():
    head = pd.read_csv(RAW, nrows=200_000)
    assert list(head.columns) == RAW_COLUMNS
    assert set(head["Payment Format"]) <= set(PAYMENT_FORMATS)
    assert set(head["Payment Currency"]) | set(head["Receiving Currency"]) <= set(CURRENCIES)
    assert set(head["Is Laundering"]) <= {0, 1}
    pd.to_datetime(head["Timestamp"], format="%Y/%m/%d %H:%M")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_ibm_aml_load.py -q`
Expected: FAIL, `ModuleNotFoundError` or `ImportError` for `src.data.ibm_aml`.

- [ ] **Step 3: Implement the constants, `load_transactions`, and the fixtures**

Read with `usecols` and explicit dtypes so the 5-million-row file stays under 3 GB while loading.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_ibm_aml_load.py tests/test_ibm_aml_real.py -q`
Expected: 4 passed and 1 skipped, or 5 passed if the raw file is present.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/data/ibm_aml.py tests/
git commit -m "Add IBM AML raw loader with schema check"
```

---

### Task 2: Bank-to-client assignment and chronological split

**Files:**
- Modify: `src/data/ibm_aml.py`
- Test: `tests/test_ibm_aml_partition.py`

**Interfaces:**
- Produces:
  - `assign_banks_to_clients(bank_tx_counts: pd.Series, n_clients: int = 10) -> dict[int, int]`. Banks in descending count (ties by ascending bank id) go one at a time to the client with the smallest running total (ties to the lowest client id).
  - `time_split(ts: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2) -> np.ndarray` of int8 (0 train, 1 val, 2 test). With `n = len(ts)`, `t1 = ts[int(n * train_frac)]`, `t2 = ts[int(n * (train_frac + val_frac))]`: split is 0 where `ts < t1`, 1 where `t1 <= ts < t2`, else 2.

- [ ] **Step 1: Write the failing tests**

```python
def test_greedy_assignment_exact():
    counts = pd.Series({1: 100, 2: 90, 3: 50, 4: 40, 5: 10})
    assert assign_banks_to_clients(counts, n_clients=2) == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}

def test_every_bank_assigned_once_and_fewer_banks_than_clients():
    counts = pd.Series({7: 5, 8: 3})
    out = assign_banks_to_clients(counts, n_clients=10)
    assert out == {7: 0, 8: 1}

def test_time_split_keeps_ties_together():
    ts = np.array([0, 1, 2, 3, 4, 5, 5, 5, 8, 9])
    assert time_split(ts).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_ibm_aml_partition.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement both functions**

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_ibm_aml_partition.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/ibm_aml.py tests/test_ibm_aml_partition.py
git commit -m "Add bank grouping and chronological split"
```

---

### Task 3: Features, history index, lazy dataset and the build CLI

**Files:**
- Create: `src/data/sequences.py`, `experiments/build_ibm_aml.py`
- Modify: `src/data/ibm_aml.py`
- Test: `tests/test_sequences.py`, `tests/test_ibm_aml_build.py`

**Interfaces:**
- Consumes: `load_transactions`, `assign_banks_to_clients`, `time_split`.
- Produces in `src/data/sequences.py`:
  - `build_history_index(entity: np.ndarray, k: int = 10) -> np.ndarray` of shape `(n, k)`, int64. Rows are assumed time-sorted. Row `j` holds the indices of the `k` most recent rows with the same entity, ending with `j` itself, left-padded with `-1`.
  - `class SequenceIndexDataset(Dataset)` with `__init__(features, history, labels, account, rows, pad_col)`. `rows` selects which table rows are examples. `__getitem__(i) -> (seq, label)` with `seq` of shape `(k, F)` float32; padded steps are all zero except `pad_col`, which is 1. `gather(idx: LongTensor) -> (seqs, labels)` does the same for a batch. Properties `example_labels` and `example_accounts` return LongTensors of length `len(rows)`.
  - `load_indexed_client(data_dir: Path, client_id: int, max_train_accounts: int | None = None, seed: int = 0) -> dict` with keys `"train"`, `"val"`, `"test"` (`SequenceIndexDataset`), `"num_features"` (int), `"n_train_accounts"` (int). `max_train_accounts` keeps a uniform random subset of whole accounts in train only.
- Produces in `src/data/ibm_aml.py`:
  - `FEATURE_NAMES: list[str]`, `NUM_FEATURES = 35`, `PAD_COL = 12`, `K_HISTORY = 10`.
  - `encode_features(df: pd.DataFrame) -> np.ndarray` of shape `(n, 35)`, float32.
  - `build_dataset(csv_path: Path, out_dir: Path, n_clients: int = 10, k: int = K_HISTORY) -> dict` (the metadata it also writes).
- On disk: `out_dir/client_{i}/table.pt` = dict of tensors `features (n,35) float32`, `history (n,10) int64` (indices into this client's table), `labels (n,) int64`, `split (n,) int8`, `account (n,) int64`, `ts (n,) int64`, rows time-sorted. `out_dir/metadata/dataset_info.json` with `"format": "indexed_v1"`, `"num_features"`, `"bank_to_client"`, `"clients": {"<id>": {"train" | "val" | "test": {"n", "fraud", "fraud_rate", "accounts"}}}` (client ids as strings), `"pooled"` in the same per-split shape, and `"fraud_rate_warning"` (bool, true when any pooled split's fraud rate is more than 5× another's).

Feature columns, in order. All "so far" features count only earlier rows of the same sending account.

| Index | Name | Definition |
|---|---|---|
| 0 | `amount_paid_log` | `log1p(amount_paid) / 20` |
| 1 | `amount_received_log` | `log1p(amount_received) / 20` |
| 2 | `same_currency` | payment currency equals receiving currency |
| 3 | `cross_bank` | sending bank differs from receiving bank |
| 4 | `self_transfer` | sending account equals receiving account |
| 5–8 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | cyclic encodings of hour of day and day of week |
| 9 | `secs_since_prev_log` | `log1p(seconds since the account's previous row) / 15`, 0 for its first row |
| 10 | `acct_tx_count_log` | `log1p(rows so far) / 10` |
| 11 | `acct_distinct_receivers_log` | `log1p(distinct receiving accounts so far) / 10` |
| 12 | `is_pad` | 0 in the table; set to 1 by the dataset on padded steps |
| 13–19 | `fmt_*` | one-hot of `pay_format` over `PAYMENT_FORMATS` |
| 20–34 | `cur_*` | one-hot of `pay_currency` over `CURRENCIES` |

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sequences.py
def test_history_index_exact():
    out = build_history_index(np.array([7, 7, 3, 7, 3]), k=3)
    assert out.tolist() == [[-1, -1, 0], [-1, 0, 1], [-1, -1, 2], [0, 1, 3], [-1, 2, 4]]

def test_dataset_pads_and_flags():
    feats = torch.arange(10, dtype=torch.float32).reshape(5, 2)   # col 1 is pad_col
    feats[:, 1] = 0
    hist = torch.tensor(build_history_index(np.array([7, 7, 3, 7, 3]), k=3))
    ds = SequenceIndexDataset(feats, hist, torch.tensor([0, 0, 1, 0, 1]),
                              torch.tensor([7, 7, 3, 7, 3]), rows=torch.tensor([2, 3]), pad_col=1)
    seq, label = ds[0]                        # table row 2: two padded steps then itself
    assert seq.tolist() == [[0, 1], [0, 1], [4, 0]] and label.item() == 1
    seqs, labels = ds.gather(torch.tensor([1]))
    assert seqs.shape == (1, 3, 2) and seqs[0, :, 0].tolist() == [0, 2, 6]
    assert ds.example_accounts.tolist() == [3, 7]

# tests/test_ibm_aml_build.py
def test_features_are_causal(tiny_csv):
    df = load_transactions(tiny_csv)
    full = encode_features(df)
    for j in (0, 57, 399):
        assert np.array_equal(encode_features(df.iloc[: j + 1])[: j + 1], full[: j + 1])

def test_features_shape_and_range(tiny_csv):
    x = encode_features(load_transactions(tiny_csv))
    assert x.shape == (400, NUM_FEATURES) and x.dtype == np.float32
    assert len(FEATURE_NAMES) == NUM_FEATURES and FEATURE_NAMES[PAD_COL] == "is_pad"
    assert np.isfinite(x).all() and np.abs(x).max() <= 5.0

def test_unknown_format_gives_zero_one_hot(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0); raw.loc[0, "Payment Format"] = "Carrier Pigeon"
    x = encode_features(load_transactions(write_raw(raw, tmp_path)))
    assert x[0, 13:20].sum() == 0 and x[1, 13:20].sum() == 1

def test_build_invariants(tiny_csv, tmp_path):
    meta = build_dataset(tiny_csv, tmp_path / "out", n_clients=3)
    tables = [torch.load(tmp_path / "out" / f"client_{i}" / "table.pt") for i in range(3)]
    assert sum(len(t["labels"]) for t in tables) == 400
    assert sum(int(t["labels"].sum()) for t in tables) == 12
    accounts = [set(t["account"].tolist()) for t in tables]
    assert not (accounts[0] & accounts[1]) and not (accounts[1] & accounts[2])
    for t in tables:
        assert (t["ts"][1:] >= t["ts"][:-1]).all()
        assert (t["split"][1:] >= t["split"][:-1]).all()          # chronological
        h = t["history"]; j = torch.arange(len(h)).unsqueeze(1)
        valid = h >= 0
        assert (h[valid] <= j.expand_as(h)[valid]).all()           # never looks ahead
        assert (t["account"][h.clamp(min=0)] == t["account"].unsqueeze(1))[valid].all()
    assert meta["format"] == "indexed_v1" and meta["num_features"] == 35
    assert "fraud_rate" in meta["clients"]["0"]["test"]
    assert meta["fraud_rate_warning"] is False

def test_fraud_concentrated_in_test_period_sets_warning(tmp_path):
    raw = make_raw(); raw["Is Laundering"] = 0
    raw.loc[[10, 390, 391, 392, 393, 394, 395, 396], "Is Laundering"] = 1
    raw.loc[250, "Is Laundering"] = 1
    meta = build_dataset(write_raw(raw, tmp_path), tmp_path / "out", n_clients=2)
    assert meta["fraud_rate_warning"] is True

def test_max_train_accounts_keeps_whole_accounts(tiny_csv, tmp_path):
    build_dataset(tiny_csv, tmp_path / "out", n_clients=2)
    c = load_indexed_client(tmp_path / "out", 0, max_train_accounts=2, seed=0)
    assert len(set(c["train"].example_accounts.tolist())) == 2 == c["n_train_accounts"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_sequences.py tests/test_ibm_aml_build.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_history_index`, `SequenceIndexDataset`, `load_indexed_client`**

- [ ] **Step 4: Implement `encode_features` and `build_dataset`**

`build_dataset` computes features and the time split once on the full frame, then slices by client and builds each client's history index over its own rows. When `fraud_rate_warning` is true it also prints the three pooled rates.

- [ ] **Step 5: Implement `experiments/build_ibm_aml.py`**

Arguments `--src` (default `datasets_capstone/ibm_aml/HI-Small_Trans.csv`), `--out` (default `data/ibm_hi_small`), `--clients` (default 10). It calls `build_dataset` and prints the per-client table from the metadata.

- [ ] **Step 6: Run to verify pass**

Run: `python3 -m pytest tests -q`
Expected: all pass (the real-file test may skip).

- [ ] **Step 7: Commit**

```bash
git add src/data experiments/build_ibm_aml.py tests/
git commit -m "Add causal per-account features, history index and IBM dataset build"
```

---

### Task 4: Fraud metrics

**Files:**
- Create: `src/utils/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `recall_at_fpr(y_true, y_prob, fpr: float = 0.01) -> float | None`: the largest recall among thresholds whose false-positive rate is at most `fpr`.
  - `best_f1_threshold(y_true, y_prob) -> float`: the score threshold that maximizes F1, where a prediction is positive when `prob >= threshold`.
  - `fraud_metrics(y_true, y_prob, threshold: float = 0.5) -> dict` with keys `n`, `n_fraud`, `pr_auc`, `roc_auc`, `recall_at_1pct_fpr`, `precision`, `recall`, `f1`, `threshold`. The three threshold-free values are `None` when only one class is present.

- [ ] **Step 1: Write the failing tests**

```python
Y = np.array([0, 0, 0, 0, 1, 1]); P = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])

def test_recall_at_fpr_exact():
    assert recall_at_fpr(Y, P, fpr=0.0) == 0.5
    assert recall_at_fpr(Y, P, fpr=0.25) == 1.0

def test_best_f1_threshold_exact():
    assert best_f1_threshold(Y, P) == pytest.approx(0.8)
    m = fraud_metrics(Y, P, threshold=0.8)
    assert m["f1"] == pytest.approx(0.8) and m["n"] == 6 and m["n_fraud"] == 2

def test_single_class_returns_none_not_error():
    m = fraud_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["recall_at_1pct_fpr"] is None
    assert m["f1"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_metrics.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the three functions with `sklearn.metrics`**

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_metrics.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/metrics.py tests/test_metrics.py
git commit -m "Add threshold-free fraud metrics"
```

---

### Task 5: Account-level DP-SGD

**Files:**
- Create: `src/federated/dp_sgd.py`
- Test: `tests/test_dp_sgd.py`

**Interfaces:**
- Consumes: `SequenceIndexDataset` (`gather`, `example_accounts`), `TemporalGraphTransformer`.
- Produces:
  - `freeze_non_dp_heads(model) -> None`: sets `requires_grad=False` on every parameter under `contrastive_head` and `aux_head`.
  - `per_example_grads(model, seqs, labels, pos_weight: float) -> dict[str, Tensor]`: one entry per trainable parameter, each of shape `(batch, *param.shape)`. The model is called with `labels=None`; the loss per example is `-(1 + (pos_weight - 1) * y) * (y * log(p) + (1 - y) * log(1 - p))` on `fraud_prob`.
  - `account_clipped_sum(grads, slot: LongTensor, n_slots: int, clip_norm: float) -> dict[str, Tensor]`: mean of each account's example gradients, each account's whole gradient scaled down to L2 norm `clip_norm` if larger, summed over accounts.
  - `noise_multiplier_for(target_epsilon: float, delta: float, sample_rate: float, total_steps: int) -> float`.
  - `@dataclass AccountDPConfig(target_epsilon: float, delta: float = 1e-6, clip_norm: float = 1.0, accounts_per_batch: int = 256, examples_per_account: int = 4)`.
  - `train_local_account_dp(model, dataset, optimizer, cfg: AccountDPConfig, sigma: float, sample_rate: float, steps: int, accountant, *, global_params=None, mu: float = 0.0, pos_weight: float = 1.0, generator=None) -> dict` with keys `loss`, `steps`, `mean_accounts_per_step`.

The algorithm, because the signatures do not determine it:

```python
# per_example_grads: trainable params are differentiated, everything else is passed through
params  = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
others  = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
buffers = dict(model.named_buffers())
def loss_one(params, x, y):
    out = functional_call(model, {**params, **others, **buffers}, (x.unsqueeze(0),), {"labels": None})
    ...
grads = vmap(grad(loss_one), in_dims=(None, 0, 0), randomness="different")(params, seqs, labels.float())

# one step of train_local_account_dp, n_acc = number of distinct accounts in dataset
picked = accounts where torch.rand(n_acc, generator=generator) < sample_rate      # Poisson
rows   = up to cfg.examples_per_account random examples of each picked account
summed = account_clipped_sum(per_example_grads(...), slot, len(picked), cfg.clip_norm)
for name, p in trainable:                                                          # also when picked is empty
    g = summed.get(name, zeros) + torch.normal(0, sigma * cfg.clip_norm, p.shape, generator=generator)
    p.grad = g / (sample_rate * n_acc)
optimizer.step()
if global_params is not None and mu > 0:                                           # decoupled FedProx
    p -= lr * mu * (p - p_global)      for each trainable p, under torch.no_grad()
accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
```

`noise_multiplier_for` wraps `opacus.accountants.utils.get_noise_multiplier(..., accountant="rdp")`. `accountant` is an `opacus.accountants.RDPAccountant` that the caller keeps per client across rounds.

- [ ] **Step 1: Write the failing tests**

```python
def small_model():
    cfg = copy.deepcopy(DEFAULT_CONFIG)          # from run_training
    torch.manual_seed(0); m = TemporalGraphTransformer(cfg); freeze_non_dp_heads(m); return m

def test_per_example_grads_match_autograd():
    m = small_model().eval()
    x = torch.randn(3, 10, 10); y = torch.tensor([0, 1, 0])
    g = per_example_grads(m, x, y, pos_weight=8.0)
    p = m(x[1:2], labels=None)["fraud_prob"][0]
    (-(8.0) * torch.log(p)).backward()
    for name, param in m.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.allclose(g[name][1], param.grad, atol=1e-6)
    assert not any(n.startswith(("contrastive_head", "aux_head")) for n in g)

def test_account_clipped_sum_exact():
    grads = {"w": torch.tensor([[3.0, 0.0], [5.0, 0.0], [0.0, 0.5]])}
    out = account_clipped_sum(grads, torch.tensor([0, 0, 1]), n_slots=2, clip_norm=1.0)
    assert torch.allclose(out["w"], torch.tensor([1.0, 0.5]))

def test_removing_one_account_moves_sum_by_at_most_clip():
    torch.manual_seed(1)
    grads = {"a": torch.randn(12, 4) * 10, "b": torch.randn(12, 3) * 10}
    slot = torch.arange(12) // 3                                   # 4 accounts
    full = account_clipped_sum(grads, slot, 4, clip_norm=1.0)
    keep = slot != 2
    rest = account_clipped_sum({k: v[keep] for k, v in grads.items()},
                               torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), 3, clip_norm=1.0)
    diff = torch.sqrt(sum(((full[k] - rest[k]) ** 2).sum() for k in full))
    assert diff <= 1.0 + 1e-5

def test_noise_multiplier_meets_target():
    q, steps = 256 / 50_000, 3906
    sigma = noise_multiplier_for(8.0, 1e-6, q, steps)
    acc = RDPAccountant()
    for _ in range(steps): acc.step(noise_multiplier=sigma, sample_rate=q)
    assert 7.5 <= acc.get_epsilon(1e-6) <= 8.0

def test_training_step_moves_only_trainable_weights_and_counts_steps(tiny_client):
    m = small_model_for(tiny_client); before = {n: p.clone() for n, p in m.named_parameters()}
    bank = {n: b.clone() for n, b in m.named_buffers()}
    acc = RDPAccountant(); opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    out = train_local_account_dp(m, tiny_client["train"], opt, AccountDPConfig(8.0, accounts_per_batch=4),
                                 sigma=1.0, sample_rate=0.5, steps=5, accountant=acc,
                                 generator=torch.Generator().manual_seed(0))
    assert out["steps"] == 5 and len(acc.history) > 0
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]) == n.startswith(("contrastive_head", "aux_head"))
    for n, b in m.named_buffers():
        assert torch.equal(b, bank[n])

def test_empty_batch_still_steps_and_adds_noise(tiny_client):
    m = small_model_for(tiny_client); before = {n: p.clone() for n, p in m.named_parameters()}
    acc = RDPAccountant(); opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    out = train_local_account_dp(m, tiny_client["train"], opt, AccountDPConfig(8.0), sigma=1.0,
                                 sample_rate=1e-9, steps=2, accountant=acc,
                                 generator=torch.Generator().manual_seed(0))
    assert out["mean_accounts_per_step"] == 0 and out["steps"] == 2
    assert any(not torch.equal(p, before[n]) for n, p in m.named_parameters() if p.requires_grad)
```

`tiny_client` is a `conftest.py` fixture: build the tiny CSV with 2 clients and return `load_indexed_client(out, 0)`. `small_model_for(client)` in `tests/helpers.py` builds the model with `num_features = client["num_features"]` and calls `freeze_non_dp_heads` on it.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_dp_sgd.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/federated/dp_sgd.py`**

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_dp_sgd.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/federated/dp_sgd.py tests/
git commit -m "Add account-level DP-SGD with per-bank Renyi accounting"
```

---

### Task 6: Experiment script with baselines and DP settings

**Files:**
- Create: `experiments/run_ibm_comparison.py`
- Test: `tests/test_run_ibm_comparison.py`

**Interfaces:**
- Consumes: `load_indexed_client`, `fraud_metrics`, `best_f1_threshold`, everything in `dp_sgd.py`; from `run_training.py`: `DEFAULT_CONFIG`, `train_local`, `federated_aggregate`, `DifferentialPrivacy`, `add_server_noise`; `TemporalGraphTransformer`.
- Produces: a CLI and `main(argv: list[str] | None = None) -> dict` that returns what it writes.

CLI arguments and defaults: `--data-path data/ibm_hi_small`, `--out results/comparison/ibm_hi_small.json`, `--models local centralized gbdt fedavg fedprox client_dp account_dp`, `--epsilons 1 3 8`, `--seeds 0`, `--rounds 10`, `--local-epochs 2`, `--batch-size 256`, `--lr 0.001`, `--mu 0.1`, `--pos-weight 8`, `--oversample-ratio 0.02`, `--delta 1e-6`, `--clip-norm 1.0`, `--dp-batch-accounts 256`, `--dp-examples-per-account 4`, `--max-train-accounts` (none).

Behaviour fixed by this plan:

- **Model config:** `DEFAULT_CONFIG` with `dataset.num_features` from the data, `training.pos_weight` from `--pos-weight`, and `training.fraud_rate_prior` set to the pooled train fraud rate.
- **Evaluation:** score every example once per split, then `fraud_metrics`. Each result holds `pooled` and `per_client` metrics for the test split. F1, precision and recall use the threshold from `best_f1_threshold` on pooled validation scores.
- **Selection:** for every model that trains in rounds or epochs, keep the weights with the best pooled validation PR-AUC and report test metrics for those weights only.
- **`local`:** one model per client, trained on its own train split for `rounds * local_epochs` epochs, evaluated on its own test split. `pooled` is computed on the concatenation of every client's test scores from its own model, so `n` is the full test count.
- **`centralized`:** one model on all clients' train data, same number of epochs.
- **`gbdt`:** `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", early_stopping=False, random_state=seed)` on the last step's features concatenated with the mean over non-padded history steps (70 columns).
- **`fedavg` / `fedprox`:** all clients take part every round; `mu` is 0 for `fedavg`. Local training uses the existing `train_local` with a weighted sampler at `--oversample-ratio`.
- **`client_dp`:** `fedprox` plus the existing client-level mechanism (`DifferentialPrivacy` clip, then `add_server_noise`), once per value in `--epsilons`. Result name `client_dp_eps{e}`.
- **`account_dp`:** `fedprox` with `train_local_account_dp`, once per value in `--epsilons`. Result name `account_dp_eps{e}`. Per client: `sample_rate = min(1, dp_batch_accounts / n_train_accounts)`, `steps_per_round = local_epochs * ceil(1 / sample_rate)`, `sigma = noise_multiplier_for(e, delta, sample_rate, rounds * steps_per_round)`, one `RDPAccountant` kept for the whole run. Abort with a clear message if `delta >= 1 / n_train_accounts` for any client. No oversampling and no server noise in this mode.
- **Seeds:** every model runs once per seed; the JSON holds each seed's result and the mean and standard deviation of the pooled metrics. Seed and client-id keys are strings. DP result names use `f"client_dp_eps{e}"` and `f"account_dp_eps{e}"` with `e` a float, so ε=8 gives `account_dp_eps8.0`.
- **JSON shape:** `{"dataset", "args", "data": {per-client and pooled counts}, "results": {name: {"seeds": {seed: {"pooled", "per_client", "privacy", "train_seconds"}}, "mean": {...}, "std": {...}}}}`. `privacy` is `null` except for `account_dp_*`, where it maps client id to `{"epsilon_spent", "sigma", "sample_rate", "n_accounts", "steps"}`.

- [ ] **Step 1: Write the failing test**

```python
def test_all_models_run_and_share_one_test_set(tiny_csv, tmp_path):
    build_dataset(tiny_csv, tmp_path / "data", n_clients=2)
    out = main(["--data-path", str(tmp_path / "data"), "--out", str(tmp_path / "r.json"),
                "--rounds", "1", "--local-epochs", "1", "--batch-size", "16",
                "--epsilons", "8", "--dp-batch-accounts", "4", "--delta", "1e-3", "--seeds", "0"])
    names = set(out["results"])
    assert names == {"local", "centralized", "gbdt", "fedavg", "fedprox",
                     "client_dp_eps8.0", "account_dp_eps8.0"}
    n_test = out["data"]["pooled"]["test"]["n"]
    for name in names:
        assert out["results"][name]["seeds"]["0"]["pooled"]["n"] == n_test
    priv = out["results"]["account_dp_eps8.0"]["seeds"]["0"]["privacy"]
    assert all(v["epsilon_spent"] <= 8.0 for v in priv.values())
    assert json.loads((tmp_path / "r.json").read_text())["results"].keys() == out["results"].keys()

def test_aborts_when_delta_too_large_for_account_count(tiny_csv, tmp_path):
    build_dataset(tiny_csv, tmp_path / "data", n_clients=2)
    with pytest.raises(SystemExit, match="delta"):
        main(["--data-path", str(tmp_path / "data"), "--out", str(tmp_path / "r.json"),
              "--models", "account_dp", "--rounds", "1", "--delta", "0.5"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_run_ibm_comparison.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `experiments/run_ibm_comparison.py`**

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_ibm_comparison.py tests/test_run_ibm_comparison.py
git commit -m "Add IBM comparison runner with baselines and both DP units"
```

---

### Task 7: Build the real dataset, run the experiments, record the gate

Needs the raw file from Task 1's prerequisite.

**Files:**
- Create: `data/ibm_hi_small/metadata/dataset_info.json` (generated), `results/comparison/ibm_hi_small_smoke.json`, `results/comparison/ibm_hi_small.json`, `results/comparison/STAGE_A0_GATE.md`
- Modify: `.gitignore` (add `!results/comparison/ibm_hi_small*.json`), `README.md`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: Confirm the schema on the real file**

Run: `python3 -m pytest tests/test_ibm_aml_real.py -q`
Expected: 1 passed. If it fails, fix the constants in `src/data/ibm_aml.py` to match the file, rerun all tests, commit.

- [ ] **Step 2: Build the dataset and check memory**

Run: `/usr/bin/time -l python3 experiments/build_ibm_aml.py`
Expected: 10 client folders under `data/ibm_hi_small/`, a printed per-client table, and "maximum resident set size" below 6,000,000,000 bytes. Record in the gate file: number of banks, transactions, fraud count, accounts and fraud per client per split, and any fraud-rate warning.

- [ ] **Step 3: Smoke run and timing**

Run: `python3 experiments/run_ibm_comparison.py --max-train-accounts 5000 --rounds 2 --local-epochs 1 --epsilons 8 --out results/comparison/ibm_hi_small_smoke.json`
Expected: finishes, every result has non-null `pr_auc`. From each result's `train_seconds`, estimate the full matrix's wall-clock time for `--max-train-accounts` of 20000, 50000 and no cap.

- [ ] **Step 4: Run the matrix**

Choose the largest of those three settings whose estimated total is under 24 hours and record the choice in the gate file. Then run:

`python3 experiments/run_ibm_comparison.py --seeds 0 1 2 --epsilons 1 3 8 [--max-train-accounts N]`

Expected: `results/comparison/ibm_hi_small.json` with all 11 results and three seeds each.

- [ ] **Step 5: Write `results/comparison/STAGE_A0_GATE.md`**

One table of mean ± std for PR-AUC, recall at 1% FPR and F1, one row per result. Then answer, with the numbers:
1. Does `fedprox` beat `local` on pooled PR-AUC, and for how many of the 10 clients?
2. What fraction of `fedprox`'s PR-AUC does `account_dp_eps8.0` keep? If under one half, the next step is the remedy list in the spec, not Stage A1.
3. How do `client_dp_eps*` and `account_dp_eps*` compare at the same ε?
4. How far is `fedprox` from `centralized` and `gbdt`?

- [ ] **Step 6: Update the docs**

Add an "IBM AML results" section to `README.md` with the table and the four answers, and a "v2 data and privacy unit" section to `docs/ARCHITECTURE.md` that summarizes decisions 1–10 of the spec. State the caveats: threshold and hyperparameter tuning are outside ε; the account cap, if one was used.

- [ ] **Step 7: Commit**

```bash
git add .gitignore README.md docs/ARCHITECTURE.md data/ibm_hi_small/metadata results/comparison
git commit -m "Record Stage A0 results on IBM AML HI-Small"
```
