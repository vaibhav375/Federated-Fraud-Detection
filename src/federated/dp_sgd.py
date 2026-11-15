"""Account-level DP-SGD inside one bank.

One account can appear in many examples (through the history window), so the
privacy unit is the account: each step samples accounts with probability q
(Poisson), averages each sampled account's example gradients, clips that
average to `clip_norm`, sums, and adds N(0, (sigma * clip_norm)^2) noise. The
Rényi accountant tracks the cost per bank across all rounds.

Per-example gradients come from torch.func on the unmodified model (Opacus's
hook-based engine does not support it); Opacus is used for accounting only.
"""
from dataclasses import dataclass

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch.func import functional_call, grad_and_value, vmap

NON_DP_HEADS = ("contrastive_head", "aux_head")


def freeze_non_dp_heads(model) -> None:
    # NT-Xent couples examples in a batch, so it has no per-account sensitivity bound.
    for name, p in model.named_parameters():
        if name.startswith(NON_DP_HEADS):
            p.requires_grad_(False)


def _per_example_grads_and_loss(model, seqs, labels, pos_weight):
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    others = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = dict(model.named_buffers())

    def loss_one(params, x, y):
        out = functional_call(model, {**params, **others, **buffers}, (x.unsqueeze(0),), {"labels": None})
        p = out["fraud_prob"].squeeze(0)
        return -(1 + (pos_weight - 1) * y) * (y * torch.log(p) + (1 - y) * torch.log(1 - p))

    return vmap(grad_and_value(loss_one), in_dims=(None, 0, 0), randomness="different")(
        params, seqs, labels.float())


def per_example_grads(model, seqs, labels, pos_weight: float) -> dict:
    return _per_example_grads_and_loss(model, seqs, labels, pos_weight)[0]


def account_clipped_sum(grads: dict, slot: torch.Tensor, n_slots: int, clip_norm: float) -> dict:
    counts = torch.bincount(slot, minlength=n_slots).clamp(min=1).float()
    means = {}
    for name, g in grads.items():
        flat = g.reshape(g.shape[0], -1)
        summed = torch.zeros(n_slots, flat.shape[1], dtype=flat.dtype).index_add_(0, slot, flat)
        means[name] = summed / counts.unsqueeze(1)
    norms = torch.sqrt(sum((m ** 2).sum(1) for m in means.values()))
    scale = torch.clamp(clip_norm / torch.clamp(norms, min=1e-12), max=1.0)
    return {name: (m * scale.unsqueeze(1)).sum(0).reshape(grads[name].shape[1:])
            for name, m in means.items()}


def noise_multiplier_for(target_epsilon: float, delta: float, sample_rate: float, total_steps: int) -> float:
    return float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                      sample_rate=sample_rate, steps=total_steps, accountant="rdp"))


@dataclass
class AccountDPConfig:
    target_epsilon: float
    delta: float = 1e-6
    clip_norm: float = 1.0
    accounts_per_batch: int = 256
    examples_per_account: int = 4
    fraud_rows_first: bool = False


def _account_groups(dataset):
    """Example positions grouped by account: (inverse, sorted positions, group starts, counts)."""
    uniq, inverse = torch.unique(dataset.example_accounts, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=len(uniq))
    starts = torch.cumsum(counts, 0) - counts
    return len(uniq), inverse, order, starts, counts


def _sample_rows(n_acc, inverse, order, starts, counts, sample_rate, per_account, generator, labels=None):
    picked = (torch.rand(n_acc, generator=generator) < sample_rate).nonzero(as_tuple=True)[0]
    if len(picked) == 0:
        return picked, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    take = torch.clamp(counts[picked], max=per_account)
    slot = torch.repeat_interleave(torch.arange(len(picked)), take)
    if labels is None:
        # Random distinct offsets within each account: rank of a random key inside the group.
        offsets = torch.cat([torch.randperm(int(c), generator=generator)[:int(t)]
                             for c, t in zip(counts[picked].tolist(), take.tolist())])
        rows = order[starts[picked].repeat_interleave(take) + offsets]
        return picked, rows, slot
    # Fraud rows first, random within each class. The choice reads only the account's own
    # rows and the account's whole contribution is clipped, so it costs no epsilon.
    chosen = []
    for s, c, t in zip(starts[picked].tolist(), counts[picked].tolist(), take.tolist()):
        group = order[s:s + c][torch.randperm(c, generator=generator)]
        chosen.append(group[torch.argsort(labels[group], descending=True, stable=True)[:t]])
    return picked, torch.cat(chosen), slot


def train_local_account_dp(model, dataset, optimizer, cfg: AccountDPConfig, sigma: float,
                           sample_rate: float, steps: int, accountant, *, global_params=None,
                           mu: float = 0.0, pos_weight: float = 1.0, generator=None) -> dict:
    model.train()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_acc, inverse, order, starts, counts = _account_groups(dataset)
    row_labels = dataset.example_labels
    denom = sample_rate * n_acc  # expected batch size, fixed so noise scale is data-independent
    lr = optimizer.param_groups[0]["lr"]
    losses, picked_counts = [], []
    for _ in range(steps):
        picked, rows, slot = _sample_rows(n_acc, inverse, order, starts, counts, sample_rate,
                                          cfg.examples_per_account, generator,
                                          labels=row_labels if cfg.fraud_rows_first else None)
        summed = {}
        if len(rows):
            seqs, labels = dataset.gather(rows)
            grads, loss = _per_example_grads_and_loss(model, seqs, labels, pos_weight)
            summed = account_clipped_sum(grads, slot, len(picked), cfg.clip_norm)
            losses.append(loss.mean().item())
        for name, p in trainable:
            noise = torch.normal(0.0, sigma * cfg.clip_norm, p.shape, generator=generator)
            p.grad = (summed.get(name, torch.zeros_like(p)) + noise) / denom
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if global_params is not None and mu > 0:
            # Decoupled FedProx: depends only on current and public global weights, so no privacy cost.
            with torch.no_grad():
                for p, g in zip(model.parameters(), global_params):
                    if p.requires_grad:
                        p.sub_(lr * mu * (p - g))
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        picked_counts.append(len(picked))
    return {"loss": sum(losses) / len(losses) if losses else None, "steps": steps,
            "mean_accounts_per_step": sum(picked_counts) / max(steps, 1)}
