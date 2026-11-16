import copy
import json

import pytest
import torch

from helpers import small_model_for

import experiments.run_ibm_comparison as runner

SMALL = ["--rounds", "1", "--local-epochs", "1", "--batch-size", "16", "--dp-batch-accounts", "4",
         "--delta", "1e-3", "--seeds", "0", "--epsilons", "8"]


def test_load_pretrained_rejects_feature_mismatch(ckpt):
    from experiments.run_dp_finetune import load_pretrained

    assert load_pretrained(ckpt, 35) is not None
    with pytest.raises(ValueError, match="35.*34|34.*35"):
        load_pretrained(ckpt, 34)


def test_embed_is_deterministic(ckpt, tiny_client):
    from experiments.run_dp_finetune import embed, load_pretrained

    backbone = load_pretrained(ckpt, 35)
    a, b = embed(backbone, tiny_client["train"]), embed(backbone, tiny_client["train"])
    assert a.shape == (len(tiny_client["train"]), 32) and torch.equal(a, b)


def test_embedding_independent_of_batch(ckpt, tiny_client):
    from experiments.run_dp_finetune import load_pretrained

    backbone = load_pretrained(ckpt, 35).eval()
    seqs, _ = tiny_client["train"].gather(torch.arange(16))
    with torch.no_grad():
        many = backbone(seqs, labels=None)["embeddings"]
        one = backbone(seqs[3:4], labels=None)["embeddings"]
    assert torch.allclose(many[3:4], one, atol=1e-6)


def test_head_model_matches_full_model_scores(ckpt, tiny_client):
    from experiments.run_dp_finetune import HeadModel, embed, load_pretrained

    backbone = load_pretrained(ckpt, 35).eval()
    seqs, _ = tiny_client["train"].gather(torch.arange(16))
    head = HeadModel(backbone.classifier, pos_weight=8.0).eval()
    with torch.no_grad():
        full = backbone(seqs, labels=None)["fraud_prob"]
        mine = head(embed(backbone, tiny_client["train"])[:16])["fraud_prob"]
    assert torch.allclose(full, mine, atol=1e-6)
    head.classifier[3].bias.data.add_(1.0)  # the head owns a copy
    assert not torch.equal(head.classifier[3].bias, backbone.classifier[3].bias)


def test_last_layer_trains_only_33_parameters(ckpt, tiny_client):
    from opacus.accountants import RDPAccountant
    from experiments.run_dp_finetune import EmbeddingDataset, HeadModel, embed, load_pretrained
    from src.federated.dp_sgd import AccountDPConfig, train_local_account_dp

    backbone = load_pretrained(ckpt, 35)
    head = HeadModel(backbone.classifier, pos_weight=8.0)
    head.set_trainable("last_layer")
    assert sum(p.numel() for p in head.parameters() if p.requires_grad) == 33
    first = head.classifier[0].weight.detach().clone()
    last = head.classifier[3].weight.detach().clone()
    ds = tiny_client["train"]
    emb = EmbeddingDataset(embed(backbone, ds), ds.example_labels, ds.example_accounts)
    opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-2)
    train_local_account_dp(head, emb, opt, AccountDPConfig(8.0, 1e-3, 1.0, 4, 4, fraud_rows_first=True),
                           1.0, 0.5, 3, RDPAccountant(), pos_weight=8.0,
                           generator=torch.Generator().manual_seed(0))
    assert torch.equal(head.classifier[0].weight, first)
    assert not torch.equal(head.classifier[3].weight, last)
    head.set_trainable("classifier")
    assert sum(p.numel() for p in head.parameters() if p.requires_grad) == 1089


def test_initial_model_can_be_selected(tiny3, monkeypatch):
    from src.data.sequences import load_indexed_client

    clients = {c: load_indexed_client(tiny3 / "data", c) for c in range(3)}
    args = runner.parse_args(["--data-path", str(tiny3 / "data"), *SMALL, "--rounds", "2"])
    build = lambda: small_model_for(clients[0])  # noqa: E731  (seeded, so every build is identical)

    def wreck(model, *a, **k):  # every round makes the model worse: scores become constant
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    monkeypatch.setattr(runner, "local_fit", wreck)
    g = torch.Generator().manual_seed(0)
    metrics, _ = runner.run_federated(args, clients, build, g, mu=0.0, include_initial=True)
    initial = runner.select_and_test([(0.0, {c: runner.snapshot(build()) for c in clients})], clients, build)
    assert metrics["selected_round"] == 0
    assert metrics["pooled"]["pr_auc"] == initial["pooled"]["pr_auc"]
    without, _ = runner.run_federated(args, clients, build, g, mu=0.0)
    assert without["selected_round"] in (1, 2)


def test_backbone_unchanged_by_finetune(ckpt, tiny3, tmp_path, monkeypatch):
    import experiments.run_dp_finetune as ft

    loaded = []
    real_load = ft.load_pretrained
    monkeypatch.setattr(ft, "load_pretrained", lambda *a, **k: loaded.append(real_load(*a, **k)) or loaded[-1])
    before = copy.deepcopy(torch.load(ckpt, weights_only=False)["state_dict"])
    out = ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "a4.json"), *SMALL])
    assert set(out["results"]) == {"zero_shot", "head_nodp", "full_nodp", "head_dp_eps8.0"}
    after = loaded[0].state_dict()  # the in-memory backbone every model and embedding was built from
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    n_test = out["data"]["pooled"]["test"]["n"]
    for name, res in out["results"].items():
        r = res["seeds"]["0"]
        assert r["pooled"]["n"] == n_test and r["pooled"]["pr_auc"] is not None, name
        assert "selected_round" in r and "val_pr_auc" in r, name
    assert out["trained_parameters"] == {"head_nodp": 1089, "head_dp": 1089, "full_nodp": out["trained_parameters"]["full_nodp"]}
    assert json.loads((tmp_path / "a4.json").read_text())["results"].keys() == out["results"].keys()


def test_dp_accounting_uses_uncapped_accounts(ckpt, tiny3, tmp_path, monkeypatch):
    import experiments.run_dp_finetune as ft

    out = ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "a4.json"),
                   *SMALL, "--models", "head_dp", "--dp-train", "last_layer", "--max-train-accounts", "5"])
    priv = out["results"]["head_dp_eps8.0"]["seeds"]["0"]["privacy"]
    assert {c: v["n_accounts"] for c, v in priv.items()} == out["data"]["train_accounts"]
    assert all(n > 5 for n in out["data"]["train_accounts"].values())
    assert out["trained_parameters"]["head_dp"] == 33

    def boom(*a, **k):
        raise AssertionError("training started before the delta check")
    monkeypatch.setattr(ft, "run_federated", boom)
    with pytest.raises(SystemExit, match="delta"):
        ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "b.json"),
                 *SMALL, "--models", "head_dp", "--delta", "0.5"])


def test_fixed_rounds_reports_last_round(tiny3, monkeypatch):
    from src.data.sequences import load_indexed_client

    clients = {c: load_indexed_client(tiny3 / "data", c) for c in range(3)}
    args = runner.parse_args(["--data-path", str(tiny3 / "data"), *SMALL, "--rounds", "2"])
    build = lambda: small_model_for(clients[0])  # noqa: E731

    def wreck(model, *a, **k):
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    monkeypatch.setattr(runner, "local_fit", wreck)
    metrics, _ = runner.run_federated(args, clients, build, torch.Generator().manual_seed(0), mu=0.0,
                                      include_initial=True, select="last")
    wrecked = build()
    wreck(wrecked)
    last = runner.select_and_test([(0.0, {c: runner.snapshot(wrecked) for c in clients})], clients, build)
    initial = runner.select_and_test([(0.0, {c: runner.snapshot(build()) for c in clients})], clients, build)
    assert metrics["selected_round"] == 2
    assert metrics["pooled"]["pr_auc"] == last["pooled"]["pr_auc"]
    assert metrics["val_selected"]["round"] == 0
    assert metrics["val_selected"]["pooled"]["pr_auc"] == initial["pooled"]["pr_auc"]


def _settings_file(tmp_path, chosen):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"chosen": chosen}))
    return str(path)


SETTINGS = {"8.0": {"lr": 0.003, "layers": "classifier", "rounds": 2},
            "nodp": {"lr": 0.001, "layers": "last_layer", "rounds": 1}}


def test_settings_applied_per_level(ckpt, tiny3, tmp_path):
    import math

    import experiments.run_dp_finetune as ft

    out = ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "a.json"),
                   *SMALL, "--seeded-dp", "--models", "head_nodp", "head_dp",
                   "--settings", _settings_file(tmp_path, SETTINGS)])
    priv = out["results"]["head_dp_eps8.0"]["seeds"]["0"]["privacy"]
    for p in priv.values():
        assert p["steps"] == 2 * math.ceil(1 / min(1.0, 4 / p["n_accounts"]))
        assert p["epsilon_spent"] <= 8.0 + 1e-6
    assert out["trained_parameters"]["head_dp_eps8.0"] == 1089
    assert out["trained_parameters"]["head_nodp"] == 33
    assert out["selection"] == "fixed_rounds" and out["settings"] == {"chosen": SETTINGS}
    assert out["results"]["head_dp_eps8.0"]["seeds"]["0"]["selected_round"] == 2
    assert out["results"]["head_nodp"]["seeds"]["0"]["selected_round"] == 1
    assert out["dp_randomness"] == "seeded"


def test_settings_missing_level_exits(ckpt, tiny3, tmp_path, monkeypatch):
    import experiments.run_dp_finetune as ft

    monkeypatch.setattr(ft, "run_federated", lambda *a, **k: pytest.fail("trained before checking settings"))
    with pytest.raises(SystemExit, match="3.0"):
        ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "a.json"),
                 *SMALL, "--epsilons", "3", "--models", "head_dp", "--settings", _settings_file(tmp_path, SETTINGS)])


def test_save_scores(ckpt, tiny3, tmp_path):
    import numpy as np

    import experiments.run_dp_finetune as ft
    from src.data.sequences import load_indexed_client

    out = ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "a.json"),
                   *SMALL, "--seeded-dp", "--models", "head_dp", "--settings", _settings_file(tmp_path, SETTINGS),
                   "--save-scores", str(tmp_path / "scores")])
    saved = np.load(tmp_path / "scores" / "head_dp_eps8.0_seed0.npz")
    assert sorted(saved.files) == ["client_0", "client_1", "client_2"]
    for c in range(3):
        assert len(saved[f"client_{c}"]) == len(load_indexed_client(tiny3 / "data", c)["test"])
    assert "test_probs" not in out["results"]["head_dp_eps8.0"]["seeds"]["0"]
