import numpy as np
import pytest
import torch

from helpers import make_raw, write_raw
from src.data.ibm_aml import build_dataset

RUN_ARGS = ["--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3"]


@pytest.fixture
def tiny3(tmp_path):
    build_dataset(write_raw(make_raw(), tmp_path), tmp_path / "data", n_clients=3)
    return tmp_path


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
