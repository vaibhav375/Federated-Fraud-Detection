import copy

import torch
from opacus.accountants import RDPAccountant

from helpers import small_model_for
from run_training import DEFAULT_CONFIG
from src.federated.dp_sgd import (AccountDPConfig, account_clipped_sum, freeze_non_dp_heads,
                                  noise_multiplier_for, per_example_grads, train_local_account_dp)
from src.models.temporal_graph_transformer import TemporalGraphTransformer

FROZEN = ("contrastive_head", "aux_head")
GOLDEN_DEFAULT_ROWS = [0, 8, 5, 9, 16, 14, 18, 17, 29, 20, 28, 27]  # recorded before fraud-rows-first existed


def small_model():
    torch.manual_seed(0)
    m = TemporalGraphTransformer(copy.deepcopy(DEFAULT_CONFIG))
    freeze_non_dp_heads(m)
    return m


def test_per_example_grads_match_autograd():
    m = small_model().eval()
    x = torch.randn(3, 10, 10)
    y = torch.tensor([0, 1, 0])
    g = per_example_grads(m, x, y, pos_weight=8.0)
    p = m(x[1:2], labels=None)["fraud_prob"][0]
    (-8.0 * torch.log(p)).backward()
    checked = 0
    for name, param in m.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.allclose(g[name][1], param.grad, atol=1e-6), name
            checked += 1
    assert checked > 0
    assert not any(n.startswith(FROZEN) for n in g)


def test_account_clipped_sum_exact():
    grads = {"w": torch.tensor([[3.0, 0.0], [5.0, 0.0], [0.0, 0.5]])}
    out = account_clipped_sum(grads, torch.tensor([0, 0, 1]), n_slots=2, clip_norm=1.0)
    assert torch.allclose(out["w"], torch.tensor([1.0, 0.5]))


def test_removing_one_account_moves_sum_by_at_most_clip():
    torch.manual_seed(1)
    grads = {"a": torch.randn(12, 4) * 10, "b": torch.randn(12, 3) * 10}
    slot = torch.arange(12) // 3  # 4 accounts
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
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=q)
    assert 7.5 <= acc.get_epsilon(1e-6) <= 8.0


def test_training_step_moves_only_trainable_weights_and_counts_steps(tiny_client):
    m = small_model_for(tiny_client)
    before = {n: p.clone() for n, p in m.named_parameters()}
    bufs = {n: b.clone() for n, b in m.named_buffers()}
    acc = RDPAccountant()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    out = train_local_account_dp(m, tiny_client["train"], opt, AccountDPConfig(8.0, accounts_per_batch=4),
                                 sigma=1.0, sample_rate=0.5, steps=5, accountant=acc,
                                 generator=torch.Generator().manual_seed(0))
    assert out["steps"] == 5 and len(acc.history) > 0 and out["mean_accounts_per_step"] > 0
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]) == n.startswith(FROZEN), n
    for n, b in m.named_buffers():
        assert torch.equal(b, bufs[n]), n


def test_empty_batch_still_steps_and_adds_noise(tiny_client):
    m = small_model_for(tiny_client)
    before = {n: p.clone() for n, p in m.named_parameters()}
    acc = RDPAccountant()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    out = train_local_account_dp(m, tiny_client["train"], opt, AccountDPConfig(8.0), sigma=1.0,
                                 sample_rate=1e-9, steps=2, accountant=acc,
                                 generator=torch.Generator().manual_seed(0))
    assert out["mean_accounts_per_step"] == 0 and out["steps"] == 2
    assert any(not torch.equal(p, before[n]) for n, p in m.named_parameters() if p.requires_grad)
    assert len(acc.history) > 0


def test_fedprox_step_pulls_toward_global(tiny_client):
    m = small_model_for(tiny_client)
    global_params = [p.detach().clone() for p in m.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(1.0)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    def dist():
        return sum(((p - g) ** 2).sum().item() for p, g in zip(m.parameters(), global_params) if p.requires_grad)
    d0 = dist()
    train_local_account_dp(m, tiny_client["train"], opt, AccountDPConfig(8.0), sigma=0.0,
                           sample_rate=1e-9, steps=1, accountant=RDPAccountant(),
                           global_params=global_params, mu=1.0, generator=torch.Generator().manual_seed(0))
    assert dist() < d0


# ---- fraud rows first inside a sampled account

def _one_account(n_rows):
    return (1, torch.zeros(n_rows, dtype=torch.long), torch.arange(n_rows),
            torch.tensor([0]), torch.tensor([n_rows]))


def test_fraud_rows_first_always_includes_the_fraud_row():
    from src.federated.dp_sgd import _sample_rows

    labels = torch.zeros(10, dtype=torch.long)
    labels[7] = 1
    g = torch.Generator().manual_seed(0)
    with_labels = sum(7 in _sample_rows(*_one_account(10), 1.0, 4, g, labels=labels)[1].tolist() for _ in range(200))
    without = sum(7 in _sample_rows(*_one_account(10), 1.0, 4, g)[1].tolist() for _ in range(200))
    assert with_labels == 200
    assert without < 150


def test_fraud_rows_first_caps_rows_per_account():
    from src.federated.dp_sgd import _sample_rows

    labels = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        picked, rows, slot = _sample_rows(*_one_account(8), 1.0, 4, g, labels=labels)
        assert len(rows) == 4 and len(set(rows.tolist())) == 4
        assert labels[rows].sum() == 4


def test_default_sampling_unchanged():
    from src.federated.dp_sgd import _sample_rows

    inverse, order = torch.arange(30) // 10, torch.arange(30)
    g = torch.Generator().manual_seed(0)
    rows = _sample_rows(3, inverse, order, torch.tensor([0, 10, 20]), torch.tensor([10, 10, 10]), 1.0, 4, g)[1]
    assert rows.tolist() == GOLDEN_DEFAULT_ROWS


def test_train_local_account_dp_fraud_first_runs(tiny_client):
    m = small_model_for(tiny_client)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    cfg = AccountDPConfig(8.0, 1e-3, 1.0, 4, 4, fraud_rows_first=True)
    out = train_local_account_dp(m, tiny_client["train"], opt, cfg, 1.0, 0.5, 3, RDPAccountant(),
                                 pos_weight=8.0, generator=torch.Generator().manual_seed(0))
    assert out["loss"] is not None and out["loss"] == out["loss"]
