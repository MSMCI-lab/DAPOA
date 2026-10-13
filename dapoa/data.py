"""Calibration data container and binary-pattern utilities.

Coding conventions (fixed throughout DAPOA):
  * ``y = 1`` marks an unauthorized candidate action, ``y = 0`` an authorized one;
  * ``D_j = 1`` means safeguard ``j`` returns BLOCK, ``D_j = 0`` ALLOW;
  * a decision vector ``d = (d_1, ..., d_m)`` is indexed by the integer ``sum_j d_j * 2**(j-1)``
    (the first component is the least significant bit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pandas is optional; CSV loading uses it when available
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def pattern_matrix(m: int) -> np.ndarray:
    """Return the (2**m, m) 0/1 matrix ``D`` with ``D[d, j]`` = bit ``j`` of ``d``."""
    return ((np.arange(1 << m)[:, None] >> np.arange(m)) & 1).astype(np.int8)


def pattern_index(decisions: np.ndarray) -> np.ndarray:
    """Map an (n, m) 0/1 decision array to integer pattern indices."""
    decisions = np.asarray(decisions, dtype=np.int64)
    return (decisions * (1 << np.arange(decisions.shape[1]))).sum(axis=1)


def pattern_string(d: int, m: int) -> str:
    """Bit string of pattern ``d`` in component order (first component first)."""
    return "".join(str((d >> j) & 1) for j in range(m))


@dataclass
class CalibrationData:
    """Binary safeguard decisions on labeled candidate actions.

    Parameters
    ----------
    decisions : (n, m) array of 0/1 (1 = BLOCK)
    y : (n,) array of 0/1 labels (1 = unauthorized)
    components : names of the m safeguards (column order)
    groups : optional (n,) array of group identifiers (users, sessions) for grouped resampling
    """

    decisions: np.ndarray
    y: np.ndarray
    components: list[str]
    groups: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    _checked: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.decisions = np.asarray(self.decisions, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.decisions.ndim != 2:
            raise ValueError("decisions must be a 2-d array (actions x safeguards)")
        if self.decisions.shape[0] != self.y.shape[0]:
            raise ValueError("decisions and y have different numbers of rows")
        if len(self.components) != self.decisions.shape[1]:
            raise ValueError("components must name every decision column")
        if not np.isin(self.decisions, (0, 1)).all():
            raise ValueError("decisions must be binary (0 = ALLOW, 1 = BLOCK)")
        if not np.isin(self.y, (0, 1)).all():
            raise ValueError("labels must be binary (0 = authorized, 1 = unauthorized)")
        if self.decisions.shape[1] < 2:
            raise ValueError("at least two safeguards are required")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
            if self.groups.shape[0] != self.y.shape[0]:
                raise ValueError("groups must have one entry per action")
        for cls in (0, 1):
            if (self.y == cls).sum() == 0:
                raise ValueError(f"no actions with label y={cls}")
        self._checked = True

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dataframe(cls, df, components: Optional[Sequence[str]] = None, label: str = "y",
                       group: Optional[str] = None, id_col: Optional[str] = None) -> "CalibrationData":
        if components is None:
            components = [c for c in df.columns if c not in (label, group, id_col)]
        dec = df[list(components)].to_numpy()
        return cls(dec, df[label].to_numpy(), list(components),
                   df[group].to_numpy() if group else None, df[id_col].to_numpy() if id_col else None)

    @classmethod
    def from_csv(cls, path: str, components: Optional[Sequence[str]] = None, label: str = "y",
                 group: Optional[str] = None, id_col: Optional[str] = None) -> "CalibrationData":
        if pd is None:  # pragma: no cover
            raise ImportError("pandas is required for CSV loading")
        return cls.from_dataframe(pd.read_csv(path), components, label, group, id_col)

    # ------------------------------------------------------------------ summaries
    @property
    def m(self) -> int:
        return self.decisions.shape[1]

    def n(self, y: int) -> int:
        return int((self.y == y).sum())

    def subset(self, y: int) -> np.ndarray:
        return self.decisions[self.y == y]

    def counts(self, y: int) -> np.ndarray:
        """Pattern counts (length 2**m) within class ``y``."""
        return np.bincount(pattern_index(self.subset(y)), minlength=1 << self.m).astype(np.int64)

    def margins(self, y: int) -> np.ndarray:
        """Empirical blocking probabilities ``p_{jy} = P_y(D_j = 1)``."""
        return self.subset(y).mean(axis=0)

    def error_counts(self) -> dict:
        """Misses (ALLOW on unauthorized) and false blocks (BLOCK on authorized) per component."""
        return {c: {"misses": int((self.subset(1)[:, j] == 0).sum()), "false_blocks": int((self.subset(0)[:, j] == 1).sum())}
                for j, c in enumerate(self.components)}

    def summary(self) -> dict:
        return {"n_actions": int(len(self.y)), "n_authorized": self.n(0), "n_unauthorized": self.n(1), "components": list(self.components),
                "n_groups": (int(len(np.unique(self.groups))) if self.groups is not None else None),
                "margins_block_y1": self.margins(1).round(6).tolist(), "margins_block_y0": self.margins(0).round(6).tolist(),
                "error_counts": self.error_counts(),
                "constant_components_y1": [c for j, c in enumerate(self.components) if self.subset(1)[:, j].std() == 0],
                "constant_components_y0": [c for j, c in enumerate(self.components) if self.subset(0)[:, j].std() == 0]}

    def split_by_group(self, n_folds: int, seed: int = 0) -> list[np.ndarray]:
        """Fold assignment (array of fold ids per action) that keeps groups together; random if no groups."""
        rng = np.random.default_rng(seed)
        if self.groups is None:
            folds = np.empty(len(self.y), dtype=np.int64)
            for cls in (0, 1):
                idx = np.flatnonzero(self.y == cls); rng.shuffle(idx); folds[idx] = np.arange(len(idx)) % n_folds
            return [folds]
        g = np.unique(self.groups); rng.shuffle(g)
        fold_of_group = {gid: i % n_folds for i, gid in enumerate(g)}
        return [np.array([fold_of_group[x] for x in self.groups])]
