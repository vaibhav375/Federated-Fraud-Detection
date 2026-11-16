import numpy as np
import torch
from scipy import stats

import experiments.run_ibm_comparison as runner
from src.data.sequences import load_indexed_client
from src.federated.dp_random import SecureDPRandom, SeededDPRandom, make_dp_random


def test_seeded_matches_torch_draws():
    rng, twin = SeededDPRandom(torch.Generator().manual_seed(3)), torch.Generator().manual_seed(3)
    assert torch.equal(rng.uniform(7), torch.rand(7, generator=twin))
    assert torch.equal(rng.permutation(9), torch.randperm(9, generator=twin))
    assert torch.equal(rng.gaussian(2.5, (3, 4)), torch.normal(0.0, 2.5, (3, 4), generator=twin))


def test_secure_gaussian_moments():
    x = SecureDPRandom().gaussian(2.0, (200_000,)).double().numpy()
    assert abs(x.mean()) < 0.02
    assert abs(x.std() - 2.0) < 0.02
    assert stats.kstest(x / 2.0, "norm").pvalue > 1e-3


def test_secure_uniform_range_and_permutation():
    rng = SecureDPRandom()
    u = rng.uniform(10_000)
    assert u.dtype == torch.float64 and float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert sorted(rng.permutation(50).tolist()) == list(range(50))
    assert rng.gaussian(1.0, (2, 3)).dtype == torch.float32


def test_secure_draws_differ_across_calls():
    rng = SecureDPRandom()
    assert not torch.equal(rng.gaussian(1.0, (100,)), rng.gaussian(1.0, (100,)))


def test_parse_args_defaults_to_secure():
    g = torch.Generator().manual_seed(0)
    args = runner.parse_args([])
    assert args.seeded_dp is False
    assert isinstance(make_dp_random(args, g), SecureDPRandom)
    assert isinstance(make_dp_random(runner.parse_args(["--seeded-dp"]), g), SeededDPRandom)


def test_account_dp_run_records_randomness(tiny3):
    from helpers import small_model_for

    data = tiny3 / "data"
    clients = {c: load_indexed_client(data, c, None, seed=0) for c in range(3)}
    build = lambda: small_model_for(clients[0])  # noqa: E731
    small = ["--data-path", str(data), "--rounds", "1", "--local-epochs", "1",
             "--dp-batch-accounts", "4", "--delta", "1e-3"]
    for extra, expected in (([], "secure"), (["--seeded-dp"], "seeded")):
        args = runner.parse_args(small + extra)
        _, privacy = runner.run_federated(args, clients, build, torch.Generator().manual_seed(0),
                                          mu=0.0, account_dp_eps=8.0)
        assert {p["dp_randomness"] for p in privacy.values()} == {expected}
