# utils/train/ordinal_ev_core.py

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

EPS = 1e-6

def enforce_monotone_nonincreasing(S: np.ndarray) -> np.ndarray:
    """
    Ensure survival columns are nonincreasing across k: S[:,0] >= S[:,1] >= ...
    S shape: (n, K) with columns ordered by thresholds k=1..K.
    """
    S = np.clip(S, EPS, 1.0 - EPS).astype(float, copy=False)
    out = S.copy()
    for j in range(1, out.shape[1]):
        out[:, j] = np.minimum(out[:, j-1], out[:, j])
    return out

def survival_to_pmf(S: np.ndarray) -> np.ndarray:
    """
    Convert survival columns S (>=1, >=2, ..., >=K) to pmf over {0,1,...,K-1,(K+)}.
    Returns pmf with shape (n, K+1), columns: [0,1,...,K-1,(K+)]
    where the last column is the tail bin (>=K).
    """
    S = enforce_monotone_nonincreasing(S)
    n, K = S.shape
    p0 = 1.0 - S[:, 0]
    mids = S[:, :-1] - S[:, 1:] if K > 1 else np.zeros((n, 0))
    p_tail = S[:, -1]
    pmf = np.column_stack([p0, mids, p_tail])
    pmf = np.clip(pmf, 0.0, 1.0)
    rowsums = pmf.sum(axis=1, keepdims=True)
    pmf = pmf / np.where(rowsums == 0.0, 1.0, rowsums)
    return pmf

def ev_from_pmf(pmf: np.ndarray, tail_mean: float, base_values: np.ndarray | None = None) -> np.ndarray:
    """
    Compute EV from pmf over [0,1,2,3,4,(5+)] style bins.
    base_values defaults to [0,1,2,3,4,tail_mean] matching pmf columns.
    """
    if base_values is None:
        base_values = np.arange(pmf.shape[1], dtype=float)
        # last column is the tail bin
        base_values[-1] = float(tail_mean)
    return pmf @ base_values

def ordinal_expected_value_from_heads(
    probas_by_k: Dict[int, np.ndarray],
    mu_tail_5plus: float,
    synth_tail_ratio: float | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
      probas_by_k: {1: S>=1, 2: S>=2, ...} arrays of shape (n,)
      mu_tail_5plus: mean value to use for the tail bin (>=5)
      synth_tail_ratio: if key 5 missing, we synthesize S>=5 = ratio * S>=4

    Returns:
      ev   : (n,) pseudo EV
      pmf  : (n, 6) over [0,1,2,3,4,(5+)]
      Sout : (n, 5) survival columns after monotone enforcement (>=1..>=5)
    """
    ks = sorted(probas_by_k.keys())
    S = np.column_stack([probas_by_k[k].reshape(-1) for k in ks])
    have_5 = (5 in ks)
    if not have_5:
        if synth_tail_ratio is None:
            # Conservative default; ideally pass a data-estimated ratio.
            synth_tail_ratio = 0.35
        S5 = np.clip(S[:, -1] * float(synth_tail_ratio), 0.0, 1.0)
        S = np.column_stack([S, S5])
    # Now S columns correspond to >=1,>=2,>=3,>=4,>=5
    S = enforce_monotone_nonincreasing(S)
    pmf = survival_to_pmf(S)
    # pmf columns: [0,1,2,3,4,(5+)]
    base_values = np.array([0,1,2,3,4,float(mu_tail_5plus)], dtype=float)
    ev = ev_from_pmf(pmf, tail_mean=float(mu_tail_5plus), base_values=base_values)
    return ev, pmf, S

def fit_tail_params_from_counts(y_counts: np.ndarray, tail_k: int = 5) -> Tuple[float, float]:
    """
    Estimate:
      tail_ratio = P(Y >= tail_k) / P(Y >= tail_k-1)
      mu_tail    = E[Y | Y >= tail_k]
    Use a large historical set of true counts.
    """
    y = np.asarray(y_counts, dtype=float)
    m4 = (y >= (tail_k - 1))
    m5 = (y >= tail_k)
    tail_ratio = (m5.mean() / m4.mean()) if m4.any() and m4.mean() > 0 else 0.0
    mu_tail = y[m5].mean() if m5.any() else float(tail_k)
    return float(tail_ratio), float(mu_tail)