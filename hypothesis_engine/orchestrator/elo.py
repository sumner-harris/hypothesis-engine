# Modified from the original work.
"""Pure Elo math — no DB, no I/O. Used by ranking agent and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloUpdate:
    elo_a_after: float
    elo_b_after: float
    expected_a: float
    k: int
    elo_a_before: float
    elo_b_before: float


def k_factor(matches_played: int, *, new_threshold: int = 5, k_new: int = 32, k_warm: int = 16) -> int:
    """Higher K for new entrants → faster convergence; lower K once seasoned."""
    return k_new if matches_played < new_threshold else k_warm


def expected_score(elo_a: float, elo_b: float, *, logistic_scale: float = 400.0) -> float:
    scale = float(logistic_scale)
    if scale <= 0:
        raise ValueError(f"logistic_scale must be positive, got {logistic_scale!r}")
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / scale))


def update_elo(
    elo_a: float,
    elo_b: float,
    winner: str,           # "a" | "b"
    matches_played_min: int,
    *,
    k_new: int = 32,
    k_warm: int = 16,
    logistic_scale: float = 400.0,
) -> EloUpdate:
    """Standard Elo update. K is decided by the *less experienced* player's count."""
    if winner not in ("a", "b"):
        raise ValueError(f"winner must be 'a' or 'b', got {winner!r}")
    k = k_factor(matches_played_min, k_new=k_new, k_warm=k_warm)
    e_a = expected_score(elo_a, elo_b, logistic_scale=logistic_scale)
    s_a = 1.0 if winner == "a" else 0.0
    delta = k * (s_a - e_a)
    return EloUpdate(
        elo_a_after=elo_a + delta,
        elo_b_after=elo_b - delta,        # zero-sum
        expected_a=e_a,
        k=k,
        elo_a_before=elo_a,
        elo_b_before=elo_b,
    )
