"""Randomness for account-level DP-SGD: Poisson account sampling, row choice and noise.

SecureDPRandom draws from the operating system's cryptographic source (os.urandom), as the
subsampling-amplification argument assumes the sampling is secret. SeededDPRandom reproduces
torch's seeded draws exactly, for tests and for the A0/A4/B1 runs made before Stage H.
"""
import os

import numpy as np
import torch


class SeededDPRandom:
    name = "seeded"

    def __init__(self, generator: torch.Generator):
        self.g = generator

    def uniform(self, n: int) -> torch.Tensor:
        return torch.rand(n, generator=self.g)

    def permutation(self, n: int) -> torch.Tensor:
        return torch.randperm(n, generator=self.g)

    def gaussian(self, std: float, shape) -> torch.Tensor:
        return torch.normal(0.0, std, tuple(shape), generator=self.g)


class SecureDPRandom:
    name = "secure"

    @staticmethod
    def _uniform_np(n: int) -> np.ndarray:
        bits = np.frombuffer(os.urandom(8 * n), dtype=np.uint64) >> np.uint64(11)
        return bits.astype(np.float64) * 2.0 ** -53  # 53 random bits: exactly representable, in [0, 1)

    def uniform(self, n: int) -> torch.Tensor:
        return torch.from_numpy(self._uniform_np(n))

    def permutation(self, n: int) -> torch.Tensor:
        return torch.from_numpy(np.argsort(self._uniform_np(n), kind="stable")).long()

    def _std_normal(self, n: int) -> np.ndarray:
        u1, u2 = self._uniform_np(n), self._uniform_np(n)
        u1 = np.where(u1 > 0, u1, 2.0 ** -53)  # log(0) guard
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)  # Box-Muller

    def gaussian(self, std: float, shape) -> torch.Tensor:
        shape = tuple(shape)
        n = int(np.prod(shape)) if shape else 1
        # Opacus secure mode: the sum of 4 standard normals over 2, against floating-point attacks.
        z = sum(self._std_normal(n) for _ in range(4)) / 2.0
        return torch.from_numpy((z * std).astype(np.float32)).reshape(shape)


def make_dp_random(args, generator: torch.Generator):
    return SeededDPRandom(generator) if getattr(args, "seeded_dp", False) else SecureDPRandom()
