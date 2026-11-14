"""Stage A0 federated training as a Flower app, with SecAgg+ aggregation.

Only trainable parameters travel. Frozen heads and the contrastive buffers stay
at the seeded initial model on every node, which is what the plain loop's
`aggregate` guarantees too. Local training is the plain loop's `local_fit`.
"""
import copy
import dataclasses
import functools
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.simulation import run_simulation
from opacus.accountants import RDPAccountant

QUANTIZATION_RANGE = 2**22
MODULUS_RANGE = 2**32
ROOT = Path(__file__).resolve().parents[2]

from experiments.run_ibm_comparison import (DEFAULT_CONFIG, account_dp_plan, local_fit,
                                            model_builder, parse_args)
from src.data.sequences import load_indexed_client
from src.federated.dp_sgd import AccountDPConfig


@dataclass
class FlowerSettings:
    data_path: str
    run_args: list = field(default_factory=list)  # argv for the A0 parse_args: one source of hyperparameters
    n_clients: int = 10
    seed: int = 0
    account_dp_eps: float | None = None
    secagg: bool = True
    clipping_range: float = 16.0
    out_dir: str = "results/flower/run"
    dump_locals: bool = False  # debug only: writes each bank's local model, which defeats SecAgg+

    @property
    def args(self):
        return parse_args(["--data-path", self.data_path, *self.run_args])


def build_model_for(settings: FlowerSettings):
    """Model builder every node shares. The prior is the pooled train fraud rate from the
    dataset metadata, which the spec treats as public."""
    meta = json.loads((Path(settings.data_path) / "metadata" / "dataset_info.json").read_text())
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset"]["num_features"] = meta["num_features"]
    config["training"]["pos_weight"] = settings.args.pos_weight
    config["training"]["fraud_rate_prior"] = meta["pooled"]["train"]["fraud"] / meta["pooled"]["train"]["n"]
    return model_builder(config)


def initial_model(build_model, seed: int):
    torch.manual_seed(seed)
    return build_model()


def trainable_arrays(model) -> list:
    return [p.detach().cpu().numpy().astype(np.float32, copy=True)
            for p in model.parameters() if p.requires_grad]


def set_trainable(model, arrays) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(arrays) != len(params):
        raise ValueError(f"expected {len(params)} arrays, got {len(arrays)}")
    with torch.no_grad():
        for p, a in zip(params, arrays):
            if tuple(a.shape) != tuple(p.shape):
                raise ValueError(f"shape {tuple(a.shape)} does not match parameter shape {tuple(p.shape)}")
            p.copy_(torch.as_tensor(np.asarray(a), dtype=p.dtype))


def round_generator(seed: int, rnd: int, client: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + rnd * 1_009 + client)


@functools.lru_cache(maxsize=1)  # one client per process: Ray actors serve every partition over a run
def _load_client(data_path: str, client_id: int, cap):
    return load_indexed_client(Path(data_path), client_id, cap, seed=0)


def client_data(settings: FlowerSettings, client_id: int) -> dict:
    return _load_client(settings.data_path, client_id, settings.args.max_train_accounts)


class BankClient(NumPyClient):
    def __init__(self, settings: FlowerSettings, client_id: int):
        self.s, self.cid = settings, client_id
        self.build = build_model_for(settings)

    def get_parameters(self, config):
        return trainable_arrays(initial_model(self.build, self.s.seed))

    def fit(self, parameters, config):
        args, rnd = self.s.args, int(config["server_round"])
        data = client_data(self.s, self.cid)
        model = initial_model(self.build, self.s.seed)
        torch.manual_seed(round_generator(self.s.seed, rnd, self.cid).initial_seed())  # own dropout stream
        set_trainable(model, parameters)
        global_params = [p.detach().clone() for p in model.parameters()]
        account_dp = None
        if self.s.account_dp_eps is not None:
            # Privacy is accounted server-side from public plan values; this accountant is a sink.
            cfg = AccountDPConfig(self.s.account_dp_eps, args.delta, args.clip_norm,
                                  args.dp_batch_accounts, args.dp_examples_per_account)
            account_dp = (cfg, account_dp_plan(args, data["n_train_accounts"], self.s.account_dp_eps),
                          RDPAccountant())
        local_fit(model, data, args, global_params, round_generator(self.s.seed, rnd, self.cid),
                  mu=args.mu, account_dp=account_dp)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # The server cannot see which bank diverged under SecAgg+ (NaN quantizes to -c), so refuse to send.
            if not torch.isfinite(p).all():
                raise ValueError(f"{name} is not finite after local training")
            # SecAgg+ clips silently to [-c, c]; a clipped value would corrupt the aggregate.
            if self.s.secagg and p.detach().abs().max().item() > self.s.clipping_range:
                raise ValueError(f"{name} exceeds the SecAgg+ clipping range {self.s.clipping_range}")
        arrays = trainable_arrays(model)
        if self.s.dump_locals:
            out = Path(self.s.out_dir) / "locals"
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"round{rnd}_client{self.cid}.npz", *arrays)
        return arrays, len(data["train"]), {}


# ---------------------------------------------------------------- server

def client_sizes(settings: FlowerSettings) -> dict:
    return {c: len(client_data(settings, c)["train"]) for c in range(settings.n_clients)}


def secagg_error_bound(n_clients: int, clipping_range: float, quantization_range: int = QUANTIZATION_RANGE) -> float:
    # One quantum (2c/Q) per client; max_weight = largest count keeps the summed weight ratio >= 1.
    return n_clients * 2 * clipping_range / quantization_range + 1e-6


def secagg_threshold(n_clients: int) -> int:
    return min(math.ceil(0.7 * n_clients), n_clients - 1)  # Flower requires threshold < num_shares


class RecordingFedAvg(FedAvg):
    """FedAvg over all clients that saves each round's aggregate and refuses partial rounds."""

    def __init__(self, settings: FlowerSettings, rounds: int):
        n = settings.n_clients
        init = trainable_arrays(initial_model(build_model_for(settings), settings.seed))
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n,
                         min_available_clients=n, accept_failures=False,
                         initial_parameters=ndarrays_to_parameters(init),
                         on_fit_config_fn=lambda r: {"server_round": r})
        self.out = Path(settings.out_dir) / "global"

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"round {server_round}: {len(failures)} client(s) failed; refusing a partial aggregate")
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self.out.mkdir(parents=True, exist_ok=True)
        np.savez(self.out / f"round{server_round}.npz", *parameters_to_ndarrays(params))
        return params, metrics


def make_server_app(settings: FlowerSettings, rounds: int) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid, context):
        strategy = RecordingFedAvg(settings, rounds)
        context = LegacyContext(context=context, config=ServerConfig(num_rounds=rounds), strategy=strategy)
        if settings.secagg:
            n = settings.n_clients
            workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(
                num_shares=n, reconstruction_threshold=secagg_threshold(n),
                max_weight=float(max(client_sizes(settings).values())),
                clipping_range=settings.clipping_range, quantization_range=QUANTIZATION_RANGE,
                modulus_range=MODULUS_RANGE))
        else:
            workflow = DefaultWorkflow()
        workflow(grid, context)

    return app


def make_client_app(settings: FlowerSettings) -> ClientApp:
    def client_fn(context):
        return BankClient(settings, int(context.node_config["partition-id"])).to_client()
    return ClientApp(client_fn=client_fn, mods=[secaggplus_mod] if settings.secagg else [])


def run(settings: FlowerSettings, rounds: int) -> None:
    out = Path(settings.out_dir).resolve()
    settings = dataclasses.replace(settings, out_dir=str(out), data_path=str(Path(settings.data_path).resolve()))
    for sub in ("global", "locals"):
        shutil.rmtree(out / sub, ignore_errors=True)
    # Ray workers import this module by reference; they inherit the driver's environment.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    run_simulation(make_server_app(settings, rounds), make_client_app(settings),
                   num_supernodes=settings.n_clients,
                   backend_config={"client_resources": {"num_cpus": 1},
                                   "init_args": {"num_cpus": 2, "include_dashboard": False,
                                                 "object_store_memory": 200 * 2**20}})
    # A halted SecAgg+ round only logs and the next round reuses stale weights, so check every round.
    missing = [r for r in range(1, rounds + 1) if not (out / "global" / f"round{r}.npz").exists()]
    if missing:
        raise RuntimeError(f"Flower produced no aggregate for round {missing[0]} (missing: {missing}); see the log above")


def aggregation_errors(out_dir, sizes: dict, rounds: int) -> list:
    """Per round: max |SecAgg+ aggregate - plain weighted average of the dumped local models|."""
    out, total = Path(out_dir), sum(sizes.values())
    errors = []
    for r in range(1, rounds + 1):
        g = np.load(out / "global" / f"round{r}.npz")
        locs = {c: np.load(out / "locals" / f"round{r}_client{c}.npz") for c in sizes}
        err = 0.0
        for k in g.files:
            plain = sum(sizes[c] / total * locs[c][k].astype(np.float64) for c in sizes)
            err = max(err, float(np.abs(g[k] - plain).max()))
        errors.append(err)
    return errors
