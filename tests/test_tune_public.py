import json

import experiments.run_dp_finetune as ft
from experiments.tune_public import choose, main

SMALL = ["--local-epochs", "1", "--batch-size", "16", "--dp-batch-accounts", "4", "--delta", "1e-3",
         "--seeded-dp"]


def _row(level, lr, layers, rounds, v):
    return {"level": level, "lr": lr, "layers": layers, "rounds": rounds, "val_pr_auc": v}


def test_choose_tie_break():
    rows = [_row("8.0", 1e-3, "classifier", 5, 0.5), _row("8.0", 1e-3, "classifier", 1, 0.5),
            _row("nodp", 1e-3, "classifier", 1, 0.4), _row("nodp", 1e-3, "last_layer", 1, 0.4),
            _row("1.0", 1e-3, "last_layer", 2, 0.3), _row("1.0", 3e-4, "last_layer", 2, 0.3),
            _row("3.0", 3e-4, "last_layer", 1, 0.1), _row("3.0", 3e-3, "classifier", 10, 0.2)]
    chosen = choose(rows)
    assert chosen["8.0"]["rounds"] == 1  # fewer rounds first
    assert chosen["nodp"]["layers"] == "last_layer"  # then the smaller head
    assert chosen["1.0"]["lr"] == 3e-4  # then the smaller learning rate
    assert chosen["3.0"] == {"lr": 3e-3, "layers": "classifier", "rounds": 10, "val_pr_auc": 0.2}  # better wins


def test_tune_public_end_to_end(ckpt, tiny3, tmp_path):
    out_path = tmp_path / "tune.json"
    out = main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(out_path),
                "--epsilons", "8", "--lrs", "1e-3", "--layers", "last_layer", "--rounds-grid", "1", "2", *SMALL])
    assert len(out["grid"]) == 4  # (eps 8, no DP) x rounds {1, 2}
    assert set(out["chosen"]) == {"8.0", "nodp"}
    assert all(set(r) == {"level", "lr", "layers", "rounds", "val_pr_auc"} for r in out["grid"])
    assert json.loads(out_path.read_text())["chosen"] == out["chosen"]
    assert out["dp_randomness"] == "seeded" and out["tuned_on"] == str(tiny3 / "data")
    ft.main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "ft.json"),
             *SMALL, "--seeds", "0", "--epsilons", "8", "--models", "head_dp", "--settings", str(out_path)])


def test_tuning_never_embeds_the_test_split(ckpt, tiny3, tmp_path, monkeypatch):
    from src.data.sequences import load_indexed_client

    test_lens = {len(load_indexed_client(tiny3 / "data", c)["test"]) for c in range(3)}
    val_lens = {len(load_indexed_client(tiny3 / "data", c)["val"]) for c in range(3)}
    seen = []
    real = ft.embed
    monkeypatch.setattr(ft, "embed", lambda b, ds: seen.append(len(ds)) or real(b, ds))
    main(["--ckpt", str(ckpt), "--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "t.json"),
          "--epsilons", "8", "--lrs", "1e-3", "--layers", "last_layer", "--rounds-grid", "1", *SMALL])
    assert len(seen) == 6  # train and val of 3 clients
    assert not (test_lens - val_lens) or not (set(seen) & (test_lens - val_lens))
