# Stage B1: Flower + SecAgg+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Stage A0 federated training (FedProx, with or without account-level DP) as a Flower ServerApp/ClientApp whose aggregation uses the SecAgg+ protocol, and prove each round's aggregate equals the plain weighted average within a stated quantization bound.

**Architecture:** The A0 runner's per-client training body is extracted into `local_fit` (no behaviour change) so the Flower client and the plain loop run the same code. `src/federated/flower_app.py` holds the client (`BankClient`), the server app (FedAvg that records each round's aggregate), and a `run()` that starts Flower's Ray simulation. Only trainable parameters travel; frozen heads and the contrastive buffers stay at the deterministic initial model on every node, which is what A0's `aggregate` already guarantees. `experiments/run_flower.py` runs it, checks each round's aggregate against the clients' debug-dumped local models, evaluates the per-round global models with the existing A0 selection code, and writes JSON.

**Tech Stack:** flwr 1.25.0 (`ServerApp`, `LegacyContext`, `DefaultWorkflow`, `SecAggPlusWorkflow`, `ClientApp`, `secaggplus_mod`, `flwr.simulation.run_simulation`), ray 2.51.1, torch 2.8, opacus 1.5.4 (accounting only). All already installed.

**Spec:** `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md` (Track B, row B1)

## Global Constraints

- No new dependencies. flwr 1.25.0 and ray 2.51.1 are installed; verify with `python3 -c "import flwr, ray; print(flwr.__version__, ray.__version__)"`.
- Machine: M1, 8 GB RAM. Ray is started with `init_args={"num_cpus": 2, "include_dashboard": False, "object_store_memory": 200 * 2**20}` and `client_resources={"num_cpus": 1}`.
- Do not run Ray simulations (Tasks 3–5) while an A0 full run (`experiments/run_ibm_comparison.py`) is alive: both compete for the same 8 GB. Check with `pgrep -f run_ibm_comparison`. Tasks 1–2 do not start Ray and may run while it is alive.
- `src/models/`, `run_training.py`, `backend/`, `frontend/` are not modified (same as A0).
- Task 1 must not change any A0 result: the A0 results JSON must stay reproducible from the branch.
- Exchanged values are the model's trainable parameters only, in `model.named_parameters()` order filtered by `requires_grad`. Buffers and frozen parameters never travel.
- SecAgg+ settings: `quantization_range = 2**22`, `modulus_range = 2**32`, `clipping_range = 16.0`, `max_weight = max(client train example counts)`, `num_shares = n_clients`, `reconstruction_threshold = ceil(0.7 * n_clients)` (7 for 10 clients; 3 shares / threshold 2 in 3-client tests, the Flower minimum).
- Aggregation error bound per round, per parameter: `n_clients * 2 * clipping_range / quantization_range + 1e-6` (because `max_weight` is the largest count, the summed weight ratio is at least 1). For 10 clients this is 7.63e-5.
- Commits are local only. Never push. Raw CSVs and `*.pt` files are never committed.
- DP noise stays seeded (`torch.Generator`) in this stage, for reproducibility. Secure noise is deferred to B2 and is stated as a limitation.

## Review Focus

1. A trainable parameter outside `[-16, 16]` is silently clipped by SecAgg+ and the aggregate is wrong without an error. Expected: the client raises before sending. Test in Task 2.
2. Flower's simulation `partition-id` not matching the A0 client id would pair one bank's data with another bank's weight. Expected: client `c` trains on `client_{c}` and reports that client's train count. Covered by Task 3's per-round check, which keys dumps and sizes by client id.
3. A client weight above `max_weight` overflows the modular sum (Flower only logs a warning). Expected: `max_weight` is the largest count, so this cannot happen. Test in Task 3 with unequal client sizes.
4. The frozen heads and contrastive buffers drifting between nodes would make the Flower model differ from A0's even with identical trainable weights. Expected: every node builds the same initial model from the seed. Test in Task 2.
5. `--debug-dump-locals` writes each bank's local model to disk, which defeats SecAgg+. Expected: it is off by default and the JSON records whether it was on. Test in Task 4.

Out of scope, stated in the gate file: client dropouts mid-round (B2 gate), client-level DP under Flower (A0's v1 baseline only), message sizes (B3).

---

### Task 1: Extract `local_fit` from the A0 runner

**Files:**
- Modify: `experiments/run_ibm_comparison.py` (`run_federated`, lines 229–289)
- Test: `tests/test_run_ibm_comparison.py`

**Interfaces:**
- Produces:
  - `account_dp_plan(args, n_accounts: int, eps: float) -> dict` with keys `sample_rate, steps_per_round, n_accounts, sigma` (the per-client dict `run_federated` builds today, minus `accountant`).
  - `local_fit(model, client: dict, args, global_params: list, generator, *, mu: float, account_dp: tuple | None = None, client_dp=None) -> None`. Trains `model` in place for one round. `account_dp` is `(AccountDPConfig, plan_dict, accountant)`; `client_dp` is the `DifferentialPrivacy` object or None. Chooses the Adam optimizer and weight decay exactly as the loop body does today.
  - `run_federated` keeps its signature and calls both.

- [ ] **Step 1: Record the golden numbers before editing.** Run on the tiny fixture and note the printed values:

```bash
python3 -W ignore - <<'EOF'
import sys; sys.path[:0] = ["tests", "."]
from pathlib import Path; import tempfile
from helpers import make_raw, write_raw
from src.data.ibm_aml import build_dataset
from experiments.run_ibm_comparison import main
d = Path(tempfile.mkdtemp()); build_dataset(write_raw(make_raw(), d), d / "out", n_clients=2)
r = main(["--data-path", str(d / "out"), "--out", str(d / "r.json"), "--models", "fedprox", "account_dp",
          "--epsilons", "8", "--rounds", "2", "--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3"])
for k in ("fedprox", "account_dp_eps8"):
    print(k, repr(r["results"][k]["mean"]["pr_auc"]), repr(r["results"][k]["mean"]["roc_auc"]))
print(repr(r["results"]["account_dp_eps8"]["seeds"]["0"]["privacy"]["0"]["epsilon_spent"]))
EOF
```

- [ ] **Step 2: Write `test_local_fit_extraction_preserves_federated_results`.** It runs the same `main([...])` call on `tiny_csv`, built with `n_clients=2`, and asserts the five printed values with `== pytest.approx(value, abs=1e-12)`. Paste the values from Step 1 as literals. Add `test_local_fit_trains_in_place`: build a small model, call `local_fit` on `tiny_client` with `mu=0.1`, and assert at least one trainable parameter changed while every frozen parameter is unchanged.
- [ ] **Step 3: Run both tests.** Run `python3 -m pytest tests/test_run_ibm_comparison.py -k "local_fit" -v`. Expected: the golden test PASSES (nothing has changed yet) and `test_local_fit_trains_in_place` FAILS with `ImportError: cannot import name 'local_fit'`.
- [ ] **Step 4: Implement `account_dp_plan` and `local_fit`.** Move the loop body and the per-client plan dict out of `run_federated`. Keep the call order, the generator use and the arguments identical. `run_federated` still owns the accountants and calls `local_fit` with the shared generator `g`.
- [ ] **Step 5: Run the full suite.** Run `python3 -m pytest -q`. Expected: all pass (36 earlier plus 2 new).
- [ ] **Step 6: Commit.**

```bash
git add experiments/run_ibm_comparison.py tests/test_run_ibm_comparison.py
git commit -m "Extract local_fit from the federated loop, results unchanged"
```

### Task 2: Flower client

**Files:**
- Create: `src/federated/flower_app.py`
- Test: `tests/test_flower_app.py`

**Interfaces:**
- Consumes: `local_fit`, `account_dp_plan` (Task 1); `model_builder`, `DEFAULT_CONFIG`, `parse_args` from the A0 runner; `load_indexed_client`.
- Produces:
  - `trainable_arrays(model) -> list[np.ndarray]` (float32 copies) and `set_trainable(model, arrays) -> None`. The latter raises `ValueError` on a count or shape mismatch.
  - `initial_model(build_model, seed: int)`: calls `torch.manual_seed(seed)` and then `build_model()`. It is the only way any node creates a model.
  - `round_generator(seed: int, rnd: int, client: int) -> torch.Generator`, seeded with `seed * 1_000_003 + rnd * 1_009 + client`.
  - `FlowerSettings` dataclass with these fields:
    - `data_path: str`
    - `run_args: list[str]` (argv for the A0 `parse_args`, so every hyperparameter has one source)
    - `n_clients: int`
    - `seed: int = 0`
    - `account_dp_eps: float | None = None`
    - `secagg: bool = True`
    - `clipping_range: float = 16.0`
    - `out_dir: str = "results/flower/run"`
    - `dump_locals: bool = False`
  - `BankClient(settings, client_id: int)`: a `flwr.client.NumPyClient` whose `fit(parameters, config)` returns `(arrays, n_train_examples, {})`. It works as follows:
    - Builds `initial_model(build_model, settings.seed)` and calls `set_trainable`.
    - Calls `local_fit` with `round_generator(seed, config["server_round"], client_id)`, `mu=args.mu`, and, when DP is on, a fresh `RDPAccountant` (accounting is done server-side in Task 4).
    - When `settings.secagg` is set, raises `ValueError` naming the parameter if any `abs()` exceeds `clipping_range`.
    - When `dump_locals` is set, writes `np.savez(out_dir/locals/round{r}_client{c}.npz, *arrays)`.
  - Client data is cached per process with `functools.lru_cache` on `(data_path, client_id, cap)`. The cap is `--max-train-accounts` from `run_args`.

- [ ] **Step 1: Write the tests** in `tests/test_flower_app.py`, with a `tiny3` fixture that calls `build_dataset(..., n_clients=3)` into `tmp_path`:
  - `test_trainable_roundtrip_leaves_frozen_and_buffers`: `set_trainable(b, trainable_arrays(a))` makes every trainable tensor equal. Every frozen parameter and buffer of `b` equals `initial_model(build, 0)`'s.
  - `test_initial_model_is_identical_across_calls`: two `initial_model(build, 0)` state dicts are equal, including the buffers.
  - `test_round_generator_repeats_and_differs`: the same `(seed, rnd, client)` gives an equal first `torch.rand(3)`. Changing `rnd` or `client` gives a different one.
  - `test_client_fit_is_deterministic`: two `BankClient(settings, 0).fit(p, {"server_round": 1})` calls return arrays that are equal elementwise, plus the client's train count.
  - `test_client_refuses_values_outside_clipping_range`: set one initial array to `20.0` with `secagg=True`. `fit` raises `ValueError` whose message contains that parameter's name.
  - `test_account_dp_client_fit_runs`: with `account_dp_eps=8.0`, `fit` returns finite arrays.
- [ ] **Step 2: Run the tests.** Run `python3 -m pytest tests/test_flower_app.py -v`. Expected: all FAIL with `ModuleNotFoundError: src.federated.flower_app`.
- [ ] **Step 3: Implement the client half of `src/federated/flower_app.py`** as specified in Interfaces.
- [ ] **Step 4: Rerun the tests.** Run `python3 -m pytest tests/test_flower_app.py -v`. Expected: 6 PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/federated/flower_app.py tests/test_flower_app.py
git commit -m "Add Flower bank client that shares A0's local training"
```

### Task 3: Server app, SecAgg+ workflow and simulation

**Files:**
- Modify: `src/federated/flower_app.py`
- Test: `tests/test_flower_app.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `client_sizes(settings) -> dict[int, int]`: train example counts per client, read from the cached client data.
  - `secagg_error_bound(n_clients: int, clipping_range: float, quantization_range: int = 2**22) -> float`, the formula in Global Constraints.
  - `RecordingFedAvg(FedAvg)` overrides `aggregate_fit`. It calls `super()`, then saves the aggregate as `np.savez(out_dir/global/round{r}.npz, *arrays)`. It uses `fraction_fit=1.0`, `fraction_evaluate=0.0`, `min_fit_clients = min_available_clients = n_clients`, `initial_parameters` set to the trainable arrays of `initial_model(build, seed)`, and `on_fit_config_fn=lambda r: {"server_round": r}`.
  - `make_server_app(settings, rounds: int) -> ServerApp`. Its `@app.main()` builds a `LegacyContext(context, ServerConfig(num_rounds=rounds), strategy)`, then runs `DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))` with the Global Constraints values when `secagg` is set, and plain `DefaultWorkflow()` otherwise.
  - `make_client_app(settings) -> ClientApp`: `client_fn(context)` reads `context.node_config["partition-id"]` as the client id. Its mods are `[secaggplus_mod]` when `secagg` is set, `[]` otherwise.
  - `run(settings, rounds: int) -> None`: wipes `out_dir/global` and `out_dir/locals`, then calls `run_simulation(make_server_app(...), make_client_app(...), num_supernodes=n_clients, backend_config={...})` with the Ray settings from Global Constraints.
  - `aggregation_errors(out_dir, sizes: dict[int, int], rounds: int) -> list[float]`: per round, the max absolute difference between `global/round{r}.npz` and the `sizes`-weighted average of the matching `locals/round{r}_client{c}.npz`.

- [ ] **Step 1: Write the tests.** Use `tiny3`, 3 clients, 2 rounds, `dump_locals=True` and `--local-epochs 1 --batch-size 32`:
  - `test_flower_fedavg_equals_plain_weighted_average`: with `secagg=False`, both errors are `<= 1e-5`, and both `global/round{1,2}.npz` exist.
  - `test_secagg_aggregate_within_quantization_bound`: with `secagg=True`, both errors are `<= secagg_error_bound(3, 16.0)` and at least one error is `> 0` (proof that SecAgg+ actually ran).
  - `test_tiny_clients_have_unequal_sizes`: `len(set(client_sizes(s).values())) > 1`, so the weighting is exercised (Review Focus 3).
  - `test_secagg_error_bound_for_ten_clients`: `secagg_error_bound(10, 16.0) == pytest.approx(7.63e-5, rel=1e-3)`.
- [ ] **Step 2: Confirm no A0 run is alive, then run the tests.** Check `pgrep -f run_ibm_comparison` (expected: no output), then run `python3 -m pytest tests/test_flower_app.py -k "flower_fedavg or secagg or unequal" -v`. Expected: FAIL with `ImportError` on `run` / `aggregation_errors`.
- [ ] **Step 3: Implement the server half and `run`** as specified in Interfaces.
- [ ] **Step 4: Rerun the tests.** Run the same command. Expected: 4 PASS. If `test_tiny_clients_have_unequal_sizes` fails, change the fixture's `make_raw` arguments (not the code) until the sizes differ, and record a ruling.
- [ ] **Step 5: Commit.**

```bash
git add src/federated/flower_app.py tests/test_flower_app.py
git commit -m "Add Flower server app with SecAgg+ aggregation and per-round check"
```

### Task 4: `experiments/run_flower.py`

**Files:**
- Create: `experiments/run_flower.py`
- Test: `tests/test_run_flower.py`

**Interfaces:**
- Consumes: Tasks 1–3; `select_and_test`, `val_pr_auc`, `evaluate_split`, `model_builder`, `account_dp_plan`, `noise_multiplier_for` from the A0 code.
- Produces: `main(argv=None) -> dict`.
  - CLI flags:
    - `--data-path`, `--out` (JSON, default `results/flower/b1.json`), `--work-dir` (default `results/flower/run`)
    - `--rounds 10`, `--seed 0`, `--account-dp-eps` (optional float), `--no-secagg`, `--debug-dump-locals`, `--clipping-range 16.0`
    - Every remaining argument goes to A0's `parse_args` as `run_args`. The default is `--max-train-accounts 20000`.
  - After `run()`, the script:
    - Loads each `global/round{r}.npz` into `initial_model` to build the per-round states.
    - Picks a model with A0's `select_and_test`.
    - When DP is on, computes privacy per client with `account_dp_plan` and an `RDPAccountant` holding `history=[(sigma, sample_rate, rounds * steps_per_round)]`.
  - JSON keys: `settings`, `secagg` (bool), `debug_dump_locals` (bool), `rounds`, `round_seconds` (list), `aggregation_errors` (list, or null when locals were not dumped), `error_bound`, `metrics` (A0 format), `privacy`.

- [ ] **Step 1: Write the tests** on `tiny3` with `--rounds 2 --local-epochs 1 --batch-size 32 --delta 1e-3`:
  - `test_run_flower_writes_metrics_and_errors`: with `--debug-dump-locals`, the JSON has 2 `aggregation_errors`, all at or below `error_bound`, and `metrics["pooled"]["pr_auc"]` is not None.
  - `test_dump_flag_recorded_and_off_by_default`: without the flag, `debug_dump_locals is False`, `aggregation_errors is None`, and no `locals/` directory exists.
  - `test_flower_account_dp_reports_same_epsilon_as_plain_loop`: `--account-dp-eps 8` with `--no-secagg`. Each client's `epsilon_spent` equals A0 `main([... "--models", "account_dp", "--epsilons", "8", same rounds and args])`'s value within `1e-9`.
- [ ] **Step 2: Run the tests.** Run `python3 -m pytest tests/test_run_flower.py -v`. Expected: FAIL, the module does not exist.
- [ ] **Step 3: Implement `experiments/run_flower.py`.**
- [ ] **Step 4: Rerun the tests, then the full suite.** Run `python3 -m pytest tests/test_run_flower.py -v`, then `python3 -m pytest -q`. Expected: 3 PASS, and the suite is all green.
- [ ] **Step 5: Commit.**

```bash
git add experiments/run_flower.py tests/test_run_flower.py
git commit -m "Add Flower+SecAgg+ experiment runner with privacy and per-round check"
```

### Task 5: Gate run and write-up

**Files:**
- Create: `results/flower/B1_GATE.md`
- Modify: `docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md` (B1 exit gate wording; the "Verified on this machine" line saying Docker is not installed)
- Modify: `README.md` (short "Flower + SecAgg+" section), `.gitignore` (`!results/flower/b1*.json`)

- [ ] **Step 1: Run the gate.** Do this on the full data when no A0 run is alive, in the background, logging to `logs/b1_gate.log`:

```bash
python3 -W ignore experiments/run_flower.py --rounds 10 --debug-dump-locals --out results/flower/b1_fedprox.json --max-train-accounts 20000
python3 -W ignore experiments/run_flower.py --rounds 10 --debug-dump-locals --account-dp-eps 8 --out results/flower/b1_account_dp_eps8.json --max-train-accounts 20000
```

Expected: every `aggregation_errors` value is `<= 7.63e-5`. If not, stop and use systematic-debugging.
- [ ] **Step 2: Write `results/flower/B1_GATE.md`.** It covers:
  - The per-round errors against the bound, and seconds per round.
  - Final pooled metrics beside A0's `fedprox` and `account_dp_eps8` rows. They are expected to be close, not identical: A0 shares one generator across clients, while B1 seeds each client and round separately.
  - The SecAgg+ parameters.
  - The out-of-scope list from Review Focus, and the seeded-noise limitation.
- [ ] **Step 3: Update the spec's B1 exit gate.** Replace "Same weights as the plain loop on a fixed seed, to float tolerance" with "Each round's SecAgg+ aggregate equals the plain weighted average of the same local models within the quantization bound n·2c/Q". Replace the Docker line with "Docker is installed".
- [ ] **Step 4: Remove the debug dumps and commit locally.** Delete `results/flower/run/` (the dumps), then commit. Do not push.

```bash
git add results/flower/B1_GATE.md results/flower/b1_*.json docs/superpowers/specs/2026-11-14-federated-fraud-v2-design.md README.md .gitignore
git commit -m "Stage B1 gate: Flower + SecAgg+ matches plain aggregation within the quantization bound"
```
