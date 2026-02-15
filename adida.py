import numpy as np
try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None

from utils import (
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _store_cs,
)
from base_forecaster import BaseForecaster
from typing import Optional, List
from conformal_intervals import ConformalIntervals
import warnings

_GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0


def _intervals_np(x: np.ndarray) -> np.ndarray:
    nonzero_idxs = np.where(x != 0)[0]
    return np.diff(nonzero_idxs + 1, prepend=0).astype(x.dtype, copy=False)


def _chunk_sums_np(array: np.ndarray, chunk_size: int) -> np.ndarray:
    n_chunks = array.size // chunk_size
    if n_chunks == 0:
        return np.empty(0, dtype=array.dtype)
    n_elems = n_chunks * chunk_size
    return array[:n_elems].reshape(n_chunks, chunk_size).sum(axis=1)


def _ses_sse_py(alpha: float, x: np.ndarray) -> float:
    complement = 1.0 - alpha
    forecast = x[0]
    sse = 0.0
    for i in range(1, x.size):
        forecast = alpha * x[i - 1] + complement * forecast
        err = x[i] - forecast
        sse += err * err
    return sse


def _ses_forecast_py(x: np.ndarray, alpha: float):
    complement = 1.0 - alpha
    fitted = np.empty_like(x)
    fitted[0] = x[0]
    for i in range(1, x.size):
        fitted[i] = alpha * x[i - 1] + complement * fitted[i - 1]
    forecast = alpha * x[-1] + complement * fitted[-1]
    fitted[0] = np.nan
    return forecast, fitted


def _expand_fitted_intervals_py(fitted: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    out[0] = np.nan
    fitted_idx = 0
    for i in range(1, y.size):
        if y[i - 1] != 0:
            fitted_idx += 1
            if fitted[fitted_idx] == 0:
                # avoid downstream division by zero
                out[i] = 1
            else:
                out[i] = fitted[fitted_idx]
        elif fitted_idx > 0:
            out[i] = out[i - 1]
        else:
            out[i] = 1
    return out


if njit is not None:
    _ses_sse_np = njit(cache=True, nogil=True)(_ses_sse_py)
    _ses_forecast_np = njit(cache=True, nogil=True)(_ses_forecast_py)
    _expand_fitted_intervals_np = njit(cache=True, nogil=True)(
        _expand_fitted_intervals_py
    )
else:
    _ses_sse_np = _ses_sse_py
    _ses_forecast_np = _ses_forecast_py
    _expand_fitted_intervals_np = _expand_fitted_intervals_py


def _bounded_minimize_scalar(fun, lower: float, upper: float, max_iter: int = 32):
    """Low-overhead bounded minimization tailored for SES alpha search."""
    left = lower
    right = upper
    c = right - _GOLDEN_RATIO * (right - left)
    d = left + _GOLDEN_RATIO * (right - left)
    fc = fun(c)
    fd = fun(d)
    for _ in range(max_iter):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - _GOLDEN_RATIO * (right - left)
            fc = fun(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + _GOLDEN_RATIO * (right - left)
            fd = fun(d)
    return 0.5 * (left + right)


def _optimized_ses_forecast_np(x: np.ndarray):
    if x.size == 1:
        return x[0], np.array([np.nan], dtype=x.dtype)
    alpha = _bounded_minimize_scalar(lambda a: _ses_sse_np(a, x), 0.1, 0.3)
    return _ses_forecast_np(x, alpha)


def _chunk_forecast_np(y: np.ndarray, aggregation_level: int):
    aggregation_level = max(1, int(aggregation_level))
    lost_remainder_data = y.size % aggregation_level
    y_cut = y[lost_remainder_data:]
    aggregation_sums = _chunk_sums_np(y_cut, aggregation_level)
    if aggregation_sums.size == 0:
        return 0.0
    sums_forecast, _ = _optimized_ses_forecast_np(aggregation_sums)
    return sums_forecast


def _adida(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
):
    y = np.asarray(ensure_float(y))
    if (y == 0).all():
        res = {"mean": np.zeros(h, dtype=y.dtype)}
        if fitted:
            fitted_vals = np.zeros_like(y)
            if fitted_vals.size:
                fitted_vals[0] = np.nan
            res["fitted"] = fitted_vals
        return res

    y_intervals = _intervals_np(y)
    if y_intervals.size:
        mean_interval = y_intervals.mean()
    else:
        mean_interval = 1.0
    aggregation_level = max(1, int(np.round(mean_interval)))

    sums_forecast = _chunk_forecast_np(y, aggregation_level)
    forecast = sums_forecast / aggregation_level
    res = {"mean": np.full(h, forecast, dtype=y.dtype)}
    if fitted:
        warnings.warn("Computing fitted values for ADIDA is very expensive")
        fitted_aggregation_levels = np.round(
            y_intervals.cumsum() / np.arange(1, y_intervals.size + 1)
        )
        fitted_aggregation_levels = _expand_fitted_intervals_np(
            np.append(np.nan, fitted_aggregation_levels), y
        )[1:].astype(np.int32)

        sums_fitted = np.empty(y.size - 1, dtype=y.dtype)
        for i, agg_lvl in enumerate(fitted_aggregation_levels):
            sums_fitted[i] = _chunk_forecast_np(y[: i + 1], int(agg_lvl))

        res["fitted"] = np.append(np.nan, sums_fitted / fitted_aggregation_levels)
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
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ):
        """Fit the ADIDA model.

        Fit an ADIDA to a time series (numpy array) `y`.

        Args:
            y (np.ndarray): Clean time series of shape (t, ).
            X (Optional[np.ndarray], optional): Optional exogenous variables. Defaults to None.

        Returns:
            ADIDA: ADIDA fitted model.
        """
        self.model_ = _adida(y=y, h=1, fitted=False)
        self._y = np.asarray(ensure_float(y))
        _store_cs(self, y=self._y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
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
        mean = np.full(h, self.model_["mean"][0], dtype=self.model_["mean"].dtype)
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
        """Access fitted ADIDA insample predictions.

        Args:
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        fitted = _adida(y=self._y, h=1, fitted=True)["fitted"]
        res = {"fitted": fitted}
        if level is not None:
            sigma = calculate_sigma(self._y - fitted, self._y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
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
        y = np.asarray(ensure_float(y))
        res = _adida(y=y, h=h, fitted=fitted)
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
    
# # Test Cases
# def test():
#     # simple increasing series
#     y = np.arange(24.0)

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
