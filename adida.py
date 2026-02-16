import numpy as np
import jax.numpy as jnp
from utils import (
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _intervals,
    _expand_fitted_intervals,
    _store_cs
)
from base_forecaster import BaseForecaster
from typing import Optional, List
from conformal_intervals import ConformalIntervals
import warnings

_ALPHA_LOWER = 0.1
_ALPHA_UPPER = 0.3
_GOLDEN_RATIO = 0.6180339887498949
_ALPHA_SEARCH_ITERS = 16


def _ensure_float_np(y):
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32)
    return y


def _interval_mean(y: np.ndarray) -> float:
    nonzero_idxs = np.flatnonzero(y != 0)
    if nonzero_idxs.size == 0:
        return 1.0
    intervals = np.diff(nonzero_idxs + 1, prepend=0)
    return float(intervals.mean())


def _ses_sse(alpha: float, x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    complement = 1.0 - alpha
    forecast = float(x[0])
    sse = 0.0
    for i in range(1, x.size):
        forecast = alpha * float(x[i - 1]) + complement * forecast
        err = float(x[i]) - forecast
        sse += err * err
    return sse


def _ses_forecast(alpha: float, x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    complement = 1.0 - alpha
    fitted_last = float(x[0])
    for i in range(1, x.size):
        fitted_last = alpha * float(x[i - 1]) + complement * fitted_last
    return alpha * float(x[-1]) + complement * fitted_last


def _optimized_ses_forecast(x: np.ndarray) -> float:
    if x.size <= 1:
        return float(x[0]) if x.size == 1 else 0.0

    left = _ALPHA_LOWER
    right = _ALPHA_UPPER
    ratio = _GOLDEN_RATIO
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = _ses_sse(c, x)
    fd = _ses_sse(d, x)

    for _ in range(_ALPHA_SEARCH_ITERS):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - ratio * (right - left)
            fc = _ses_sse(c, x)
        else:
            left = c
            c = d
            fc = fd
            d = left + ratio * (right - left)
            fd = _ses_sse(d, x)

    alpha = 0.5 * (left + right)
    return _ses_forecast(alpha, x)


def _chunk_forecast_fast(y: np.ndarray, aggregation_level: int) -> float:
    aggregation_level = max(1, int(aggregation_level))
    lost_remainder_data = y.size % aggregation_level
    y_cut = y[lost_remainder_data:]
    n_chunks = y_cut.size // aggregation_level
    if n_chunks == 0:
        return 0.0

    aggregation_sums = y_cut[: n_chunks * aggregation_level].reshape(
        n_chunks, aggregation_level
    ).sum(axis=1)
    if n_chunks == 1:
        return float(aggregation_sums[0])
    return _optimized_ses_forecast(aggregation_sums)


def _repeat_val_np(val: float, h: int, dtype):
    return np.full((h,), val, dtype=dtype)


def _adida_point(
    y,  # time series
    h: int,  # forecasting horizon
):
    y = _ensure_float_np(y)
    mean_interval = _interval_mean(y)
    aggregation_level = max(1, int(round(mean_interval)))
    sums_forecast = _chunk_forecast_fast(y, aggregation_level)
    forecast = sums_forecast / aggregation_level
    return {"mean": _repeat_val_np(val=forecast, h=h, dtype=y.dtype)}


def _adida(
    y,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
):
    if not fitted:
        return _adida_point(y=y, h=h)

    y = ensure_float(y)
    if (y == 0).all():
        res = {"mean": jnp.zeros(h, dtype=y.dtype)}
        fitted_vals = jnp.zeros_like(y)
        if fitted_vals.size > 0:
            fitted_vals = fitted_vals.at[0].set(jnp.nan)
        res["fitted"] = fitted_vals
        return res

    point_res = _adida_point(y=np.asarray(y), h=h)
    res = {"mean": jnp.asarray(point_res["mean"], dtype=y.dtype)}
    warnings.warn("Computing fitted values for ADIDA is very expensive")

    y_intervals = _intervals(y)
    fitted_aggregation_levels = jnp.round(
        y_intervals.cumsum() / jnp.arange(1, y_intervals.size + 1)
    )
    fitted_aggregation_levels = _expand_fitted_intervals(
        jnp.append(jnp.nan, fitted_aggregation_levels), y
    )[1:].astype(jnp.int32)

    y_np = np.asarray(y)
    sums_fitted = []
    for i, agg_lvl in enumerate(fitted_aggregation_levels):
        agg_lvl_int = max(1, int(agg_lvl))
        sums_fitted.append(_chunk_forecast_fast(y_np[: i + 1], agg_lvl_int))
    sums_fitted = jnp.asarray(sums_fitted, dtype=y.dtype)

    res["fitted"] = jnp.append(jnp.nan, sums_fitted / fitted_aggregation_levels)
    return res


class ADIDA(BaseForecaster):
    def __init__(
        self,
        alias: str = "ADIDA",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        """ADIDA model.

        Aggregate-Dissagregate Intermittent Demand Approach: Uses temporal aggregation to reduce the
        number of zero observations. Once the data has been agregated, it uses the optimized SES to
        generate the forecasts at the new level. It then breaks down the forecast to the original
        level using equal weights.

        ADIDA specializes on sparse or intermittent series are series with very few non-zero observations.
        They are notoriously hard to forecast, and so, different methods have been developed
        especifically for them.

        References:
            - [Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., & Assimakopoulos, V. (2011). An aggregate–disaggregate intermittent demand approach (ADIDA) to forecasting: an empirical proposition and analysis. Journal of the Operational Research Society, 62(3), 544-554.](https://researchportal.bath.ac.uk/en/publications/an-aggregate-disaggregate-intermittent-demand-approach-adida-to-f).

        Args:
            alias (str, optional): Custom name of the model. Defaults to "ADIDA".
            prediction_intervals (Optional[ConformalIntervals], optional): Information to compute conformal prediction intervals.
                By default, the model will compute the native prediction intervals. Defaults to None.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True
        self.conformal_params = prediction_intervals

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        """Fit the ADIDA model.

        Fit an ADIDA to a time series (numpy array) `y`.

        Args:
            y (np.ndarray): Clean time series of shape (t, ).
            X (Optional[np.ndarray], optional): Optional exogenous variables. Defaults to None.

        Returns:
            ADIDA: ADIDA fitted model.
        """
        y_np = _ensure_float_np(y)
        self.model_ = _adida_point(y=y_np, h=1)
        self._y = y_np
        if self.prediction_intervals is not None:
            y_jax = ensure_float(y_np)
            self._y = y_jax
            _store_cs(self, y=y_jax, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        """Predict with fitted ADIDA.

        Args:
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if level is not None and self.prediction_intervals is None:
            raise Exception(
                "You have to instantiate the class with `prediction_intervals`"
                "to calculate them"
            )

        mean_arr = np.asarray(self.model_["mean"])
        mean = _repeat_val_np(val=mean_arr[0], h=h, dtype=mean_arr.dtype)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        res["mean"] = jnp.asarray(res["mean"])
        res = _add_predict_conformal_intervals(self, res, level)
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        """Access fitted ADIDA insample predictions.

        Args:
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        fitted = _adida(y=self._y, h=1, fitted=True)["fitted"]
        res = {"fitted": fitted}
        if level is not None:
            y_jax = ensure_float(self._y)
            sigma = calculate_sigma(y_jax - fitted, y_jax.size)
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
        """Memory Efficient ADIDA predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (np.ndarray): Clean time series of shape (n,).
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional insample exogenous of shape (t, n_x). Defaults to None.
            X_future (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.
            fitted (bool, optional): Whether or not to return insample predictions. Defaults to False.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if level is not None and self.prediction_intervals is None:
            raise Exception(
                "You have to instantiate the class with `prediction_intervals`"
                "to calculate them"
            )

        if fitted:
            res = _adida(y=y, h=h, fitted=True)
        else:
            y_np = _ensure_float_np(y)
            res = _adida_point(y=y_np, h=h)
        if level is None:
            return res
        level = sorted(level)
        y_jax = ensure_float(y)
        res["mean"] = jnp.asarray(res["mean"])
        res = _add_conformal_intervals(self, fcst=res, y=y_jax, X=X, level=level)
        if fitted:
            sigma = calculate_sigma(y_jax - res["fitted"], y_jax.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res
    
# # Test Cases
# def test():
#     # simple increasing series
#     y = jnp.arange(24.0)

#     ci = ConformalIntervals(method="conformal_distribution")
#     model = ADIDA(prediction_intervals=ci)

#     fitted_model = model.fit(y)

#     result = fitted_model.predict(h=12, level=[60, 75])
#     forecast = fitted_model.forecast(y, h=12, level=[80, 95])

#     # ---- Assertions ----
#     assert "mean" in result, "Missing mean forecast"
#     assert len(result["mean"]) == 12, "Forecast length mismatch"

#     for lvl in [60, 75]:
#         assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
#         assert len(result[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(result[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

#     for lvl in [80, 95]:
#         assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
#         assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

#     print("Test passed!")


# if __name__ == "__main__":
#     test()
