"""Hidden tests of K.M.1, slice S5 (spec §13.4, §13.5, §35 S5, §T6.1).

One test function per check:
- ``test_normalises_nontrivially`` (HT.NONTRIV);
- ``test_shape_and_index_valid`` (HT.SHAPE);
- ``test_numeric_valid`` (HT.FINITE);
- ``test_leak_free`` (HT.LEAK, the §13.5 leakage oracle).

The solution module ``normaliser`` is imported from ``/work`` (``PYTHONPATH``)
inside a fixture, so an import failure is reported per test. The case panel
and its burn-in are read from ``data/`` in the working directory. Every seed
is fixed here, so it is part of the ``hidden_tests`` asset (§13.5).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RTOL = 1e-9
ATOL = 1e-12
N_CUTS = 5
AFFINE_SEED = 20261301
NOISE_SEED = 20261302
SHUFFLE_SEED = 20261303

Normalise = Callable[[pd.DataFrame], pd.DataFrame]


def _data_dir() -> Path:
    return Path.cwd() / "data"


def _load_panel() -> pd.DataFrame:
    with np.load(_data_dir() / "panel.npz", allow_pickle=False) as z:
        dates = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
        assets = [str(a) for a in z["assets"]]
        columns = [str(c) for c in z["columns"]]
        values = np.asarray(z["values"], dtype=float)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    return pd.DataFrame(
        values.reshape(len(dates) * len(assets), len(columns)),
        index=index,
        columns=columns,
    )


def _burn_in() -> int:
    meta = json.loads((_data_dir() / "meta.json").read_text(encoding="utf-8"))
    return int(meta["burn_in"])


@pytest.fixture(scope="module")
def normalise() -> Normalise:
    return importlib.import_module("normaliser").normalise


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _load_panel()


def _call(normalise: Normalise, panel: pd.DataFrame) -> pd.DataFrame:
    out = normalise(panel.copy())
    assert isinstance(out, pd.DataFrame), "normalise must return a DataFrame"
    return out


def _close(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(
        np.allclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
    )


def test_normalises_nontrivially(normalise: Normalise, panel: pd.DataFrame) -> None:
    out = _call(normalise, panel).reindex(index=panel.index, columns=panel.columns)
    assert not _close(out, panel), "the output equals the input"
    assets = panel.index.get_level_values(1)
    unique = pd.Index(pd.unique(assets))
    rng = np.random.default_rng(AFFINE_SEED)
    a = rng.uniform(0.5, 2.0, size=(len(unique), panel.shape[1]))
    b = rng.uniform(-5.0, 5.0, size=(len(unique), panel.shape[1]))
    pos = unique.get_indexer(assets)
    transformed = pd.DataFrame(
        panel.to_numpy(dtype=float) * a[pos] + b[pos],
        index=panel.index,
        columns=panel.columns,
    )
    out_t = _call(normalise, transformed).reindex(
        index=panel.index, columns=panel.columns
    )
    assert _close(out, out_t), "the output changes under a positive affine transform"


def test_shape_and_index_valid(normalise: Normalise, panel: pd.DataFrame) -> None:
    out = _call(normalise, panel)
    assert out.index.equals(panel.index), "index differs from the input's"
    assert list(out.columns) == list(panel.columns), "columns differ from the input's"


def test_numeric_valid(normalise: Normalise, panel: pd.DataFrame) -> None:
    out = _call(normalise, panel).reindex(index=panel.index, columns=panel.columns)
    position = panel.groupby(level=1, sort=False).cumcount().to_numpy()
    after_burn_in = position >= _burn_in()
    values = out.to_numpy(dtype=float)[after_burn_in]
    assert np.isfinite(values).all(), "non-finite value after the burn-in"


def test_leak_free(normalise: Normalise, panel: pd.DataFrame) -> None:
    dates = panel.index.get_level_values(0)
    unique = np.sort(pd.unique(dates))
    n = len(unique)
    assert n >= 6, "the leakage fixture needs at least 6 unique dates"
    before = _call(normalise, panel)
    before_dates = before.index.get_level_values(0)
    noise_rng = np.random.default_rng(NOISE_SEED)
    shuffle_rng = np.random.default_rng(SHUFFLE_SEED)
    values = panel.to_numpy(dtype=float)
    scale = np.nanstd(values, axis=0)
    for k in range(1, N_CUTS + 1):
        cut = unique[(k * n) // 6 - 1]
        future = np.asarray(dates > cut)
        perturbed = values.copy()
        perturbed[future] += noise_rng.standard_normal((int(future.sum()), values.shape[1])) * scale
        frame = pd.DataFrame(perturbed, index=panel.index, columns=panel.columns)
        order = shuffle_rng.permutation(int(future.sum()))
        frame = pd.concat([frame[~future], frame[future].iloc[order]])
        after = _call(normalise, frame)
        b = before[np.asarray(before_dates <= cut)]
        a = after[np.asarray(after.index.get_level_values(0) <= cut)]
        assert a.index.equals(b.index), f"index at or before cut point {k} differs"
        assert list(a.columns) == list(b.columns), f"columns differ at cut point {k}"
        assert _close(a, b), f"outputs at or before cut point {k} depend on later data"
