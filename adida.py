import jax.numpy as jnp
from jax import jit
from functools import lru_cache
from utils import (
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _intervals,
    _chunk_forecast,
    _expand_fitted_intervals,
    _store_cs
)
from base_forecaster import BaseForecaster
from typing import Optional, List
from conformal_intervals import ConformalIntervals
import warnings

def _interval_mean(y: jnp.ndarray) -> jnp.ndarray:
    y_intervals = _intervals(y)
    valid_count = jnp.count_nonzero(y_intervals)
    return jnp.where(valid_count > 0, jnp.sum(y_intervals) / valid_count, 1.0)


@lru_cache(maxsize=256)
def _get_chunk_forecast_runner(aggregation_level: int):
    aggregation_level = max(1, int(aggregation_level))

    compiled = jit(lambda series: _chunk_forecast(series, aggregation_level))
    state = {"use_jit": True}

    def _run(series: jnp.ndarray):
        # Cache the decision so fallback costs are paid only once.
        if state["use_jit"]:
            try:
                return compiled(series)
            except Exception:
                state["use_jit"] = False
        return _chunk_forecast(series, aggregation_level)

    return _run


def _chunk_forecast_fast(y: jnp.ndarray, aggregation_level: int):
    return _get_chunk_forecast_runner(aggregation_level)(y)


def _repeat_val_jax(val: jnp.ndarray, h: int):
    return jnp.full((h,), val)


def _adida_point(
    y: jnp.ndarray,  # time series
    h: int,  # forecasting horizon
):
    mean_interval = _interval_mean(y)
    aggregation_level = max(1, int(jnp.round(mean_interval).item()))
    sums_forecast = _chunk_forecast_fast(y, aggregation_level)
    forecast = sums_forecast / aggregation_level
    return {"mean": _repeat_val_jax(val=forecast, h=h)}


def _adida(
    y: jnp.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
):
    if not fitted:
        return _adida_point(y=y, h=h)

    if (y == 0).all():
        res = {"mean": jnp.zeros(h, dtype=y.dtype)}
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
        """Predict with fitted ADIDA.

        Args:
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = _repeat_val_jax(val=self.model_["mean"][0], h=h)
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
