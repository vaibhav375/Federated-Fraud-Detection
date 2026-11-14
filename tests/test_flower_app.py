import numpy as np
import pytest
import torch

RUN_ARGS = ["--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3"]


def _settings(tiny3, **kw):
    from src.federated.flower_app import FlowerSettings
    return FlowerSettings(data_path=str(tiny3 / "data"), run_args=RUN_ARGS, n_clients=3,
                          out_dir=str(tiny3 / "run"), **kw)


def test_trainable_roundtrip_leaves_frozen_and_buffers(tiny3):
    from src.federated.flower_app import build_model_for, initial_model, set_trainable, trainable_arrays
    build = build_model_for(_settings(tiny3))
    a = initial_model(build, 0)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(1.0)
    b = initial_model(build, 0)
    set_trainable(b, trainable_arrays(a))
    ref = initial_model(build, 0)
    for (n, pb), pa, pr in zip(b.named_parameters(), a.parameters(), ref.parameters()):
        assert torch.equal(pb, pa if pb.requires_grad else pr), n
    for (n, bb), br in zip(b.named_buffers(), ref.buffers()):
        assert torch.equal(bb, br), n


def test_set_trainable_rejects_wrong_shapes(tiny3):
    from src.federated.flower_app import build_model_for, initial_model, set_trainable, trainable_arrays
    m = initial_model(build_model_for(_settings(tiny3)), 0)
    arrays = trainable_arrays(m)
    with pytest.raises(ValueError):
        set_trainable(m, arrays[:-1])
    with pytest.raises(ValueError):
        set_trainable(m, [np.zeros((1,), np.float32)] + arrays[1:])


def test_initial_model_is_identical_across_calls(tiny3):
    from src.federated.flower_app import build_model_for, initial_model
    build = build_model_for(_settings(tiny3))
    s1, s2 = initial_model(build, 0).state_dict(), initial_model(build, 0).state_dict()
    assert s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)


def test_round_generator_repeats_and_differs():
    from src.federated.flower_app import round_generator
    draw = lambda *k: torch.rand(3, generator=round_generator(*k))  # noqa: E731
    assert torch.equal(draw(0, 1, 2), draw(0, 1, 2))
    assert not torch.equal(draw(0, 1, 2), draw(0, 2, 2))
    assert not torch.equal(draw(0, 1, 2), draw(0, 1, 1))


def _initial_arrays(settings):
    from src.federated.flower_app import build_model_for, initial_model, trainable_arrays
    return trainable_arrays(initial_model(build_model_for(settings), settings.seed))


def test_client_fit_is_deterministic(tiny3):
    from src.federated.flower_app import BankClient, client_data
    s = _settings(tiny3)
    p = _initial_arrays(s)
    out1, n1, _ = BankClient(s, 0).fit(p, {"server_round": 1})
    out2, n2, _ = BankClient(s, 0).fit(p, {"server_round": 1})
    assert n1 == n2 == len(client_data(s, 0)["train"])
    assert all(np.array_equal(x, y) for x, y in zip(out1, out2))
    assert any(not np.array_equal(x, y) for x, y in zip(out1, p))


def test_client_refuses_values_outside_clipping_range(tiny3):
    from src.federated.flower_app import BankClient, build_model_for, initial_model
    s = _settings(tiny3, secagg=True)
    p = _initial_arrays(s)
    p[0] = np.full_like(p[0], 20.0)
    first = next(n for n, q in initial_model(build_model_for(s), 0).named_parameters() if q.requires_grad)
    with pytest.raises(ValueError, match=first):
        BankClient(s, 0).fit(p, {"server_round": 1})


def test_account_dp_client_fit_runs(tiny3):
    from src.federated.flower_app import BankClient
    s = _settings(tiny3, account_dp_eps=8.0)
    out, _, _ = BankClient(s, 1).fit(_initial_arrays(s), {"server_round": 1})
    assert all(np.isfinite(x).all() for x in out)


def test_client_dumps_locals_only_when_asked(tiny3):
    from src.federated.flower_app import BankClient
    s = _settings(tiny3)
    BankClient(s, 2).fit(_initial_arrays(s), {"server_round": 1})
    assert not (tiny3 / "run" / "locals").exists()
    s = _settings(tiny3, dump_locals=True)
    out, _, _ = BankClient(s, 2).fit(_initial_arrays(s), {"server_round": 1})
    saved = np.load(tiny3 / "run" / "locals" / "round1_client2.npz")
    assert all(np.array_equal(saved[f"arr_{i}"], x) for i, x in enumerate(out))


# ---------------------------------------------------------------- server + simulation (Ray)

def _simulate(tiny3, secagg, rounds=2):
    from src.federated.flower_app import aggregation_errors, client_sizes, run
    s = _settings(tiny3, secagg=secagg, dump_locals=True)
    run(s, rounds)
    return s, aggregation_errors(s.out_dir, client_sizes(s), rounds)


def test_flower_fedavg_equals_plain_weighted_average(tiny3):
    s, errors = _simulate(tiny3, secagg=False)
    assert len(errors) == 2 and max(errors) <= 1e-5
    assert all((tiny3 / "run" / "global" / f"round{r}.npz").exists() for r in (1, 2))


def test_secagg_aggregate_within_quantization_bound(tiny3):
    from src.federated.flower_app import secagg_error_bound
    s, errors = _simulate(tiny3, secagg=True)
    assert len(errors) == 2 and max(errors) <= secagg_error_bound(3, 16.0)
    assert max(errors) > 0  # quantization noise proves the SecAgg+ path ran


def test_tiny_clients_have_unequal_sizes(tiny3):
    from src.federated.flower_app import client_sizes
    assert len(set(client_sizes(_settings(tiny3)).values())) > 1


def test_secagg_error_bound_for_ten_clients():
    from src.federated.flower_app import secagg_error_bound
    assert secagg_error_bound(10, 16.0) == pytest.approx(10 * 32 / 2**22 + 1e-6, rel=1e-9)  # 7.73e-5


def test_secagg_threshold_is_below_shares():
    from src.federated.flower_app import secagg_threshold
    assert secagg_threshold(10) == 7 and secagg_threshold(3) == 2


def test_server_refuses_to_aggregate_with_failures(tiny3):
    from src.federated.flower_app import RecordingFedAvg
    strategy = RecordingFedAvg(_settings(tiny3), rounds=1)
    with pytest.raises(RuntimeError, match="1 client"):
        strategy.aggregate_fit(1, [], [RuntimeError("boom")])
