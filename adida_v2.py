import warnings
from functools import lru_cache
from typing import Dict, List, Optional

import jax.numpy as jnp
from jax import jit, lax, vmap

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
from utils import (
    _add_conformal_intervals,
    _add_fitted_pi,
    _add_predict_conformal_intervals,
    _expand_fitted_intervals,
    _intervals,
    _store_cs,
    calculate_sigma,
    ensure_float,
)

_ALPHA_LOWER = 0.1
_ALPHA_UPPER = 0.3
_ALPHA_GRID_SIZE = 12
_REFINE_GRID_SIZE = 5


@jit
def _mean_interval_and_nonzero_count(y: jnp.ndarray):
    """Compute ADIDA interval mean with static-shape JAX ops."""
    nonzero_mask = y != 0
    nonzero_count = jnp.count_nonzero(nonzero_mask).astype(jnp.int32)

    # nonzero indices with fixed output shape for jit-friendliness
    idx = jnp.nonzero(nonzero_mask, size=y.shape[0], fill_value=0)[0] + 1
    prev_idx = jnp.concatenate((jnp.array([0], dtype=idx.dtype), idx[:-1]))
    intervals = idx - prev_idx

    valid = (jnp.arange(y.shape[0], dtype=idx.dtype) < nonzero_count).astype(y.dtype)
    interval_sum = (intervals.astype(y.dtype) * valid).sum()

    mean_interval = jnp.where(
        nonzero_count > 0,
        interval_sum / nonzero_count.astype(y.dtype),
        jnp.array(1.0, dtype=y.dtype),
    )
    return mean_interval, nonzero_count


def _ses_sse(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    complement = 1.0 - alpha
    init = (x[0], x[0], jnp.array(0.0, dtype=x.dtype))

    def step(carry, obs):
        forecast, prev_obs, sse = carry
        forecast = alpha * prev_obs + complement * forecast
        err = obs - forecast
        return (forecast, obs, sse + err * err), None

    (_, _, sse), _ = lax.scan(step, init, x[1:])
    return sse


def _ses_one_step(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    complement = 1.0 - alpha
    init = (x[0], x[0])

    def step(carry, obs):
        forecast, prev_obs = carry
        forecast = alpha * prev_obs + complement * forecast
        return (forecast, obs), None

    (fitted_last, last_obs), _ = lax.scan(step, init, x[1:])
    return alpha * last_obs + complement * fitted_last


@jit
def _optimized_ses_forecast(x: jnp.ndarray) -> jnp.ndarray:
    alpha_grid = jnp.linspace(
        jnp.array(_ALPHA_LOWER, dtype=x.dtype),
        jnp.array(_ALPHA_UPPER, dtype=x.dtype),
        _ALPHA_GRID_SIZE,
        dtype=x.dtype,
    )
    sse_grid = vmap(lambda a: _ses_sse(a, x))(alpha_grid)
    best_idx = jnp.argmin(sse_grid)

    left_idx = jnp.maximum(best_idx - 1, 0)
    right_idx = jnp.minimum(best_idx + 1, _ALPHA_GRID_SIZE - 1)

    refine_grid = jnp.linspace(
        alpha_grid[left_idx],
        alpha_grid[right_idx],
        _REFINE_GRID_SIZE,
        dtype=x.dtype,
    )
    refine_sse = vmap(lambda a: _ses_sse(a, x))(refine_grid)
    best_alpha = refine_grid[jnp.argmin(refine_sse)]

    return _ses_one_step(best_alpha, x)


@lru_cache(maxsize=512)
def _get_chunk_forecast_runner(aggregation_level: int):
    m = max(1, int(aggregation_level))

    @jit
    def _run(series: jnp.ndarray):
        n = series.shape[0]
        lost_remainder_data = n % m
        y_cut = series[lost_remainder_data:]
        n_chunks = y_cut.shape[0] // m

        if n_chunks == 0:
            return jnp.array(0.0, dtype=series.dtype)

        chunk_sums = y_cut[: n_chunks * m].reshape((n_chunks, m)).sum(axis=1)
        if n_chunks == 1:
            return chunk_sums[0]

        return _optimized_ses_forecast(chunk_sums)

    return _run


def _chunk_forecast_fast(y: jnp.ndarray, aggregation_level: int) -> jnp.ndarray:
    return _get_chunk_forecast_runner(aggregation_level)(y)


def _adida_point(
    y: jnp.ndarray,
    h: int,
):
    mean_interval, nonzero_count = _mean_interval_and_nonzero_count(y)
    if int(nonzero_count.item()) == 0:
        return {
            "mean": jnp.zeros(h, dtype=y.dtype),
            "aggregation_level": 1,
        }

    aggregation_level = max(1, int(jnp.round(mean_interval).item()))
    sums_forecast = _chunk_forecast_fast(y, aggregation_level)
    point_forecast = sums_forecast / aggregation_level
    return {
        "mean": jnp.full((h,), point_forecast, dtype=y.dtype),
        "aggregation_level": aggregation_level,
    }


def _adida(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
):
    if not fitted:
        return _adida_point(y=y, h=h)

    if (y == 0).all():
        res = {
            "mean": jnp.zeros(h, dtype=y.dtype),
            "aggregation_level": 1,
        }
        fitted_vals = jnp.zeros_like(y)
        if fitted_vals.size > 0:
            fitted_vals = fitted_vals.at[0].set(jnp.nan)
        res["fitted"] = fitted_vals
        return res

    res = _adida_point(y=y, h=h)
    warnings.warn("Computing fitted values for ADIDA is very expensive")

    y_intervals = _intervals(y)
    fitted_aggregation_levels = jnp.round(
        y_intervals.cumsum() / jnp.arange(1, y_intervals.size + 1)
    )
    fitted_aggregation_levels = _expand_fitted_intervals(
        jnp.append(jnp.nan, fitted_aggregation_levels), y
    )[1:].astype(jnp.int32)

    sums_fitted = []
    for i, agg_lvl in enumerate(fitted_aggregation_levels):
        agg_lvl_int = max(1, int(agg_lvl))
        sums_fitted.append(_chunk_forecast_fast(y[: i + 1], agg_lvl_int))
    sums_fitted = jnp.asarray(sums_fitted, dtype=y.dtype)

    res["fitted"] = jnp.append(jnp.nan, sums_fitted / fitted_aggregation_levels)
    return res


def _group_indices_by_level(levels: List[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for i, lvl in enumerate(levels):
        groups.setdefault(lvl, []).append(i)
    return groups


def _adida_batch_point(y: jnp.ndarray, h: int):
    """Batch-first ADIDA point forecasts grouped by aggregation level."""
    if y.ndim != 2:
        raise ValueError("`y` must have shape (n_series, n_time)")

    mean_intervals, nonzero_counts = vmap(_mean_interval_and_nonzero_count)(y)
    agg_levels = jnp.maximum(1, jnp.round(mean_intervals).astype(jnp.int32))
    levels_host = [int(v) for v in agg_levels.tolist()]
    groups = _group_indices_by_level(levels_host)

    point = jnp.zeros((y.shape[0],), dtype=y.dtype)
    for lvl, idxs in groups.items():
        idx = jnp.asarray(idxs, dtype=jnp.int32)
        runner = _get_chunk_forecast_runner(lvl)
        sums = vmap(runner)(y[idx])
        point = point.at[idx].set(sums / lvl)

    point = jnp.where(nonzero_counts > 0, point, jnp.zeros_like(point))
    mean = jnp.broadcast_to(point[:, None], (point.shape[0], h))
    return {"mean": mean, "aggregation_level": agg_levels}


class ADIDA(BaseForecaster):
    def __init__(
        self,
        alias: str = "ADIDA",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True
        self.conformal_params = prediction_intervals

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        y = ensure_float(y)
        self._y = y
        self.model_ = _adida_point(y=y, h=1)
        _store_cs(self, y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        mean = jnp.full((h,), self.model_["mean"][0], dtype=self.model_["mean"].dtype)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_predict_conformal_intervals(self, res, level)
        else:
            raise Exception(
                "You have to instantiate the class with `prediction_intervals`"
                "to calculate them"
            )
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        fitted = _adida(y=self._y, h=1, fitted=True)["fitted"]
        res = {"fitted": fitted}
        if level is not None:
            sigma = calculate_sigma(self._y - fitted, self._y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        y = ensure_float(y)
        if fitted:
            res = _adida(y=y, h=h, fitted=True)
        else:
            res = _adida_point(y=y, h=h)
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        else:
            raise Exception(
                "You have to instantiate the class with `prediction_intervals`"
                "to calculate them"
            )
        if fitted:
            sigma = calculate_sigma(y - res["fitted"], y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res

    def forecast_batch(
        self,
        y: jnp.ndarray,
        h: int,
    ):
        y = ensure_float(y)
        return _adida_batch_point(y=y, h=h)


ADIDAv2 = ADIDA
