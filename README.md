"""
forecast.py
───────────
End-to-end forecasting pipeline for M5-style retail demand data.

Models   : ETS (Holt-Winters), SARIMA, Prophet
Metrics  : RMSE, MAE, MASE, Bias
Output   : ForecastResult dataclass with actuals, forecasts, metrics, CI bands
"""

from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

warnings.filterwarnings("ignore")

TEST_DAYS = 90          # hold-out window
SEASONAL_PERIOD = 7     # weekly


# ── Data helpers ─────────────────────────────────────────────────────────────────

def load_series(
    path: str = "data/m5_sample.csv",
    store_id: str = "CA_1",
    category_id: str = "FOODS",
) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    mask = (df["store_id"] == store_id) & (df["category_id"] == category_id)
    s = df[mask].set_index("date")["sales"].sort_index()
    s = s.asfreq("D").fillna(0)
    return s


def split(series: pd.Series, test_days: int = TEST_DAYS):
    return series.iloc[:-test_days], series.iloc[-test_days:]


# ── Metrics ───────────────────────────────────────────────────────────────────────

def _rmse(a, f):  return float(np.sqrt(np.mean((a - f) ** 2)))
def _mae(a, f):   return float(np.mean(np.abs(a - f)))
def _bias(a, f):  return float(np.mean(f - a))          # + = over-forecast

def _mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray) -> float:
    naive_mae = np.mean(np.abs(np.diff(train)))
    return float(np.mean(np.abs(actual - forecast)) / naive_mae) if naive_mae else np.nan

def _smape(a, f):
    denom = (np.abs(a) + np.abs(f)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0, np.abs(a - f) / denom)
    return float(np.mean(ratio) * 100)

def compute_metrics(actual: pd.Series, forecast: pd.Series, train: pd.Series) -> dict:
    a, f, tr = actual.values, forecast.values, train.values
    return {
        "RMSE":  round(_rmse(a, f),  2),
        "MAE":   round(_mae(a, f),   2),
        "MASE":  round(_mase(a, f, tr), 3),
        "sMAPE": round(_smape(a, f), 2),
        "Bias":  round(_bias(a, f),  2),
    }


# ── Models ────────────────────────────────────────────────────────────────────────

def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=horizon, freq="D"
    )


def fit_ets(train: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (forecast, lower_ci, upper_ci)."""
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)

    sim = model.simulate(horizon, repetitions=200, error="add")
    idx = _future_index(train, horizon)
    fc  = pd.Series(model.forecast(horizon).clip(0).values, index=idx)
    lo  = pd.Series(np.percentile(sim, 10, axis=1).clip(0), index=idx)
    hi  = pd.Series(np.percentile(sim, 90, axis=1).clip(0), index=idx)
    return fc, lo, hi


def fit_sarima(train: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """SARIMA(1,1,1)(1,0,1)[7] – fast but solid."""
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast_obj = model.get_forecast(steps=horizon)
    idx  = _future_index(train, horizon)
    fc   = pd.Series(forecast_obj.predicted_mean.clip(0).values, index=idx)
    ci   = forecast_obj.conf_int(alpha=0.20)
    lo   = pd.Series(ci.iloc[:, 0].clip(0).values, index=idx)
    hi   = pd.Series(ci.iloc[:, 1].clip(0).values, index=idx)
    return fc, lo, hi


def fit_prophet(train: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    df_p = train.reset_index().rename(columns={"date": "ds", "sales": "y"})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.80,
        mcmc_samples=0,
    )
    m.fit(df_p)

    future  = m.make_future_dataframe(periods=horizon)
    pred    = m.predict(future).set_index("ds").iloc[-horizon:]
    idx     = _future_index(train, horizon)
    fc  = pd.Series(pred["yhat"].clip(0).values, index=idx)
    lo  = pd.Series(pred["yhat_lower"].clip(0).values, index=idx)
    hi  = pd.Series(pred["yhat_upper"].clip(0).values, index=idx)
    return fc, lo, hi


# ── Result container ──────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    series:    pd.Series
    train:     pd.Series
    test:      pd.Series
    store_id:  str
    cat_id:    str
    forecasts: dict[str, pd.Series]             = field(default_factory=dict)
    lower:     dict[str, pd.Series]             = field(default_factory=dict)
    upper:     dict[str, pd.Series]             = field(default_factory=dict)
    metrics:   dict[str, dict[str, float]]      = field(default_factory=dict)

    def best_model(self, metric: str = "RMSE") -> str:
        return min(self.metrics, key=lambda m: self.metrics[m].get(metric, np.inf))


# ── Pipeline entry point ──────────────────────────────────────────────────────────

MODEL_FNS = {
    "ETS":    fit_ets,
    "SARIMA": fit_sarima,
    "Prophet": fit_prophet,
}

def run_pipeline(
    data_path:   str = "data/m5_sample.csv",
    store_id:    str = "CA_1",
    category_id: str = "FOODS",
    test_days:   int = TEST_DAYS,
) -> ForecastResult:

    series = load_series(data_path, store_id, category_id)
    train, test = split(series, test_days)
    result = ForecastResult(series=series, train=train, test=test,
                            store_id=store_id, cat_id=category_id)

    for name, fn in MODEL_FNS.items():
        try:
            fc, lo, hi = fn(train, len(test))
            result.forecasts[name] = fc
            result.lower[name]     = lo
            result.upper[name]     = hi
            result.metrics[name]   = compute_metrics(test, fc, train)
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")

    return result


if __name__ == "__main__":
    from src.generate_data import generate
    generate()
    res = run_pipeline()
    print("\n── Metrics ─────────────────────────────")
    for model, m in res.metrics.items():
        print(f"  {model:8s}  RMSE={m['RMSE']:7.1f}  MAE={m['MAE']:7.1f}  "
              f"MASE={m['MASE']:.3f}  sMAPE={m['sMAPE']:.1f}%  Bias={m['Bias']:+.1f}")
    print(f"\n  Best model (RMSE): {res.best_model()}")
