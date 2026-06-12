"""
dashboard.py
────────────
Interactive Demand Forecasting Dashboard
  → python src/dashboard.py
  → http://127.0.0.1:8050

Features
  • Store & category selector (15 combinations)
  • Model toggle + best-model highlight
  • Forecast chart with 80% confidence bands
  • Residuals + error distribution panel
  • KPI cards (RMSE / MAE / MASE / sMAPE / Bias)
  • Full metrics table with conditional formatting
  • Dark-on-light professional theme
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc

from src.generate_data import generate
from src.forecast import run_pipeline, MODEL_FNS

# ── Bootstrap app ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Demand Forecasting · M5 Walmart",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ── Constants ─────────────────────────────────────────────────────────────────────
STORES    = ["CA_1", "TX_1", "WI_1"]
CATS      = ["FOODS", "HOBBIES", "HOUSEHOLD", "SNAP_FOODS", "CLOTHING"]
MODELS    = list(MODEL_FNS.keys())
PALETTE   = {"ETS": "#E07B39", "SARIMA": "#2196F3", "Prophet": "#9C27B0"}
ACTUAL_C  = "#1A2F4A"
METRIC_COLS = ["RMSE", "MAE", "MASE", "sMAPE", "Bias"]

DATA_PATH = "data/m5_sample.csv"
if not os.path.exists(DATA_PATH):
    print("[INFO] Generating data...")
    generate(DATA_PATH)

# ── Layout helpers ────────────────────────────────────────────────────────────────

def kpi_card(label, value, color, icon):
    return dbc.Card(
        dbc.CardBody([
            html.Div(icon, style={"fontSize": "22px", "marginBottom": "4px"}),
            html.P(label, className="text-muted mb-1", style={"fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.08em", "textTransform": "uppercase"}),
            html.H4(str(value), style={"color": color, "fontWeight": "700", "margin": "0"}),
        ]),
        className="shadow-sm text-center",
        style={"borderTop": f"3px solid {color}", "borderRadius": "10px"},
    )


def section_header(title, subtitle=""):
    return html.Div([
        html.H5(title, style={"margin": "0", "fontWeight": "700", "color": "#1A2F4A"}),
        html.Small(subtitle, className="text-muted") if subtitle else None,
    ], style={"marginBottom": "12px"})


# ── App layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#F4F6FB", "minHeight": "100vh", "padding": "28px"}, children=[

    # ── Header ──
    dbc.Row(dbc.Col(html.Div([
        html.H2("📦 Demand Forecasting Dashboard", style={"margin": "0", "fontWeight": "800", "color": "#1A2F4A", "fontSize": "28px"}),
        html.P("M5 Walmart Retail Sales · ETS  ·  SARIMA  ·  Prophet  ·  90-day forecast horizon",
               className="text-muted mb-0", style={"fontSize": "14px"}),
    ])), class_name="mb-4"),

    # ── Control bar ──
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Store", style={"fontWeight": "600", "fontSize": "12px"}),
            dcc.Dropdown(id="store-dd",
                options=[{"label": s, "value": s} for s in STORES], value="CA_1",
                clearable=False, style={"fontSize": "14px"}),
        ], width=2),
        dbc.Col([
            html.Label("Category", style={"fontWeight": "600", "fontSize": "12px"}),
            dcc.Dropdown(id="cat-dd",
                options=[{"label": c, "value": c} for c in CATS], value="FOODS",
                clearable=False, style={"fontSize": "14px"}),
        ], width=2),
        dbc.Col([
            html.Label("Models", style={"fontWeight": "600", "fontSize": "12px"}),
            dcc.Checklist(id="model-chk",
                options=[{"label": f"  {m}", "value": m} for m in MODELS],
                value=MODELS, inline=True,
                inputStyle={"marginRight": "4px", "marginLeft": "14px"},
                style={"fontSize": "14px", "paddingTop": "6px"}),
        ], width=4),
        dbc.Col([
            html.Label("Show CI bands", style={"fontWeight": "600", "fontSize": "12px"}),
            dcc.RadioItems(id="ci-toggle",
                options=[{"label": "  On", "value": "on"}, {"label": "  Off", "value": "off"}],
                value="on", inline=True,
                inputStyle={"marginRight": "4px", "marginLeft": "14px"},
                style={"fontSize": "14px", "paddingTop": "6px"}),
        ], width=2),
        dbc.Col([
            html.Label("History window", style={"fontWeight": "600", "fontSize": "12px"}),
            dcc.Dropdown(id="zoom-dd",
                options=[{"label": "Last 6 months", "value": 180},
                         {"label": "Last year", "value": 365},
                         {"label": "All data", "value": 9999}],
                value=180, clearable=False, style={"fontSize": "14px"}),
        ], width=2),
    ])), class_name="shadow-sm mb-4", style={"borderRadius": "12px"}),

    # ── KPI cards (populated by callback) ──
    html.Div(id="kpi-row", className="mb-4"),

    # ── Main forecast chart ──
    dbc.Card(dbc.CardBody([
        section_header("Forecast vs Actuals", "Shaded area = 80% prediction interval"),
        dcc.Loading(dcc.Graph(id="forecast-chart", config={"displayModeBar": False},
                              style={"height": "420px"})),
    ]), class_name="shadow-sm mb-4", style={"borderRadius": "12px"}),

    # ── Residuals + error dist ──
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            section_header("Forecast Residuals", "Actual − Forecast over test period"),
            dcc.Graph(id="residual-chart", config={"displayModeBar": False},
                      style={"height": "300px"}),
        ]), class_name="shadow-sm h-100", style={"borderRadius": "12px"}), width=7),

        dbc.Col(dbc.Card(dbc.CardBody([
            section_header("Error Distribution", "Histogram of daily forecast errors"),
            dcc.Graph(id="error-hist", config={"displayModeBar": False},
                      style={"height": "300px"}),
        ]), class_name="shadow-sm h-100", style={"borderRadius": "12px"}), width=5),
    ], class_name="mb-4"),

    # ── Metrics table ──
    dbc.Card(dbc.CardBody([
        section_header("Model Performance Summary", "Lower RMSE/MAE/MASE = better   |   Bias: positive = over-forecast"),
        dash_table.DataTable(
            id="metrics-tbl",
            columns=[{"name": c, "id": c} for c in ["Model"] + METRIC_COLS],
            style_header={
                "backgroundColor": "#1A2F4A", "color": "white",
                "fontWeight": "700", "fontSize": "12px", "textAlign": "center",
            },
            style_cell={"textAlign": "center", "padding": "10px", "fontSize": "14px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#F8F9FB"},
            ],
        ),
    ]), class_name="shadow-sm mb-4", style={"borderRadius": "12px"}),

    # ── Footer ──
    html.Hr(),
    html.P("Khalid Said Yusuf · MSc Operations & Supply Chain Analytics · Aarhus University · 2025",
           className="text-muted text-center", style={"fontSize": "12px"}),

    # ── Cache store ──
    dcc.Store(id="result-store"),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────────

@callback(
    Output("result-store", "data"),
    Input("store-dd", "value"),
    Input("cat-dd", "value"),
)
def run_models(store, cat):
    res = run_pipeline(DATA_PATH, store_id=store, category_id=cat)
    payload = {
        "series": res.series.reset_index().rename(columns={"index": "date"}).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
        "train_end": str(res.train.index[-1].date()),
        "test_start": str(res.test.index[0].date()),
        "forecasts": {m: fc.reset_index().rename(columns={"index": "date"}).assign(date=lambda d: d["date"].astype(str)).to_dict("records") for m, fc in res.forecasts.items()},
        "lower":     {m: s.values.tolist() for m, s in res.lower.items()},
        "upper":     {m: s.values.tolist() for m, s in res.upper.items()},
        "metrics":   res.metrics,
        "best":      res.best_model(),
    }
    return payload


@callback(
    Output("kpi-row", "children"),
    Output("forecast-chart", "figure"),
    Output("residual-chart", "figure"),
    Output("error-hist", "figure"),
    Output("metrics-tbl", "data"),
    Output("metrics-tbl", "style_data_conditional"),
    Input("result-store", "data"),
    Input("model-chk", "value"),
    Input("ci-toggle", "value"),
    Input("zoom-dd", "value"),
)
def update_all(payload, selected, ci_on, zoom):
    if not payload:
        empty = go.Figure()
        return [], empty, empty, empty, [], []

    # Re-hydrate series
    series = pd.DataFrame(payload["series"]).set_index("date")["sales"]
    series.index = pd.to_datetime(series.index)
    train_end  = pd.Timestamp(payload["train_end"])
    test_start = pd.Timestamp(payload["test_start"])
    metrics    = payload["metrics"]
    best       = payload["best"]

    # Build forecast series
    forecasts, lowers, uppers = {}, {}, {}
    for m in MODELS:
        if m not in payload["forecasts"]:
            continue
        fc_df = pd.DataFrame(payload["forecasts"][m]).set_index("date")
        fc_df.index = pd.to_datetime(fc_df.index)
        forecasts[m] = fc_df.iloc[:, 0]
        lo = pd.Series(payload["lower"][m], index=fc_df.index)
        hi = pd.Series(payload["upper"][m], index=fc_df.index)
        lowers[m], uppers[m] = lo, hi

    test_actuals = series[series.index >= test_start]

    # ── KPI cards ──
    kpis = []
    for m in selected:
        if m not in metrics:
            continue
        mx = metrics[m]
        star = " ★" if m == best else ""
        col = PALETTE[m]
        kpis.append(dbc.Col(kpi_card(f"{m}{star}", f"RMSE {mx['RMSE']}", col, "📊"), width="auto"))

    kpi_row = dbc.Row(kpis, class_name="g-3")

    # ── Forecast chart ──
    cutoff = series.index[-1] - pd.Timedelta(days=zoom)
    s_zoom = series[series.index >= cutoff]
    train_zoom = s_zoom[s_zoom.index <= train_end]

    fig = go.Figure()

    # Training background
    fig.add_vrect(
        x0=str(train_zoom.index[0].date()), x1=str(train_end.date()),
        fillcolor="rgba(26,47,74,0.06)", layer="below", line_width=0,
        annotation_text="Train", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#888",
    )
    # Test background
    fig.add_vrect(
        x0=str(test_start.date()), x1=str(series.index[-1].date()),
        fillcolor="rgba(224,123,57,0.06)", layer="below", line_width=0,
        annotation_text="Test", annotation_position="top right",
        annotation_font_size=10, annotation_font_color="#888",
    )

    fig.add_trace(go.Scatter(
        x=s_zoom.index, y=s_zoom.values,
        name="Actual", line=dict(color=ACTUAL_C, width=2), opacity=0.9,
    ))

    for m in selected:
        if m not in forecasts:
            continue
        fc = forecasts[m]
        col = PALETTE[m]
        is_best = m == best

        if ci_on == "on":
            fig.add_trace(go.Scatter(
                x=list(uppers[m].index) + list(lowers[m].index[::-1]),
                y=list(uppers[m].values) + list(lowers[m].values[::-1]),
                fill="toself", fillcolor=col.replace(")", ",0.12)").replace("rgb", "rgba") if col.startswith("rgb") else col + "1f",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            name=f"{m}{' ★' if is_best else ''}",
            line=dict(color=col, width=2.5 if is_best else 1.8, dash="dash" if not is_best else "solid"),
        ))

    fig.add_vline(x=str(test_start.date()), line_dash="dot", line_color="#888", line_width=1.5)

    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(l=50, r=20, t=20, b=60),
        yaxis_title="Units Sold", xaxis_title=None,
        font=dict(family="Segoe UI, sans-serif", size=13),
    )

    # ── Residuals chart ──
    fig_res = go.Figure()
    for m in selected:
        if m not in forecasts:
            continue
        resid = test_actuals.values - forecasts[m].values
        fig_res.add_trace(go.Bar(
            x=test_actuals.index, y=resid,
            name=m, marker_color=PALETTE[m], opacity=0.75,
        ))
    fig_res.add_hline(y=0, line_color="#333", line_width=1)
    fig_res.update_layout(
        template="plotly_white", barmode="group", hovermode="x unified",
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", y=-0.25),
        yaxis_title="Error (units)", xaxis_title=None,
        font=dict(family="Segoe UI, sans-serif", size=12),
    )

    # ── Error histogram ──
    fig_hist = go.Figure()
    for m in selected:
        if m not in forecasts:
            continue
        resid = test_actuals.values - forecasts[m].values
        fig_hist.add_trace(go.Histogram(
            x=resid, name=m, marker_color=PALETTE[m], opacity=0.65,
            nbinsx=20, histnorm="probability density",
        ))
    fig_hist.add_vline(x=0, line_dash="dot", line_color="#333", line_width=1.5)
    fig_hist.update_layout(
        template="plotly_white", barmode="overlay",
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", y=-0.3),
        xaxis_title="Residual (units)", yaxis_title="Density",
        font=dict(family="Segoe UI, sans-serif", size=12),
    )

    # ── Metrics table ──
    tbl_data = []
    for m in MODELS:
        if m not in metrics:
            continue
        row = {"Model": f"{m} ★" if m == best else m}
        row.update(metrics[m])
        tbl_data.append(row)

    best_rmse = min(metrics[m]["RMSE"] for m in metrics)
    style_cond = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#F8F9FB"},
        {"if": {"filter_query": f"{{RMSE}} = {best_rmse}"},
         "backgroundColor": "#E8F5E9", "color": "#2E7D32", "fontWeight": "700"},
    ]

    return kpi_row, fig, fig_res, fig_hist, tbl_data, style_cond


if __name__ == "__main__":
    print("Starting dashboard → http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
