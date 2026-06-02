# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import os
import json
import pickle
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kyushu River Forecast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.02em; }
.stApp { background-color: #0a0f1e; color: #e8eaf0; }

section[data-testid="stSidebar"] {
    background-color: #0d1528;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8eaf0;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.finding-box {
    background: #0d1f3c;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #c8d0e0;
    line-height: 1.6;
}
.warning-box {
    background: #1a1500;
    border-left: 3px solid #ffa726;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #ffe082;
    line-height: 1.6;
}
div[data-testid="stMetric"] {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="stMetric"] label {
    color: #7a8fa6 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    color: #4fc3f7 !important;
    font-size: 1.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STATIONS = {
    "2590100": {"name": "SENOSHITA",    "river": "CHIKUGO GAWA",  "area": 2315.0, "lat": 33.3161, "lon": 130.4972},
    "2590101": {"name": "ARASE",        "river": "CHIKUGO GAWA",  "area": 1443.0, "lat": 33.3433, "lon": 130.8350},
    "2590200": {"name": "HINODE-BASHI", "river": "ONGA GAWA",     "area":  695.0, "lat": 33.7500, "lon": 130.7300},
    "2590210": {"name": "MUTABE",       "river": "MATSUURA GAWA", "area":  275.0, "lat": 33.3600, "lon": 130.0100},
    "2590220": {"name": "TAMANA",       "river": "KIKUCHI GAWA",  "area":  906.0, "lat": 32.9400, "lon": 130.5900},
    "2590230": {"name": "ONOBUCHI",     "river": "SENDAI GAWA",   "area": 1348.0, "lat": 31.8600, "lon": 130.3400},
    "2590300": {"name": "YOKOISHI",     "river": "KUMA GAWA",     "area": 1856.0, "lat": 32.4600, "lon": 130.6600},
    "2590301": {"name": "HITOYOSHI",    "river": "KUMA GAWA",     "area": 1137.0, "lat": 32.2100, "lon": 130.7700},
    "2590400": {"name": "KASHIWADA",    "river": "OYODO GAWA",    "area": 2126.0, "lat": 31.9500, "lon": 131.4000},
    "2590401": {"name": "HIWATASHI",    "river": "OYODO GAWA",    "area":  860.6, "lat": 31.8600, "lon": 131.1000},
}

TOPOLOGY = {
    "2590101": "2590100",
    "2590301": "2590300",
    "2590401": "2590400",
}

RIVER_COLORS = {
    "CHIKUGO GAWA":  "#4fc3f7",
    "ONGA GAWA":     "#66bb6a",
    "MATSUURA GAWA": "#ef5350",
    "KIKUCHI GAWA":  "#ab47bc",
    "SENDAI GAWA":   "#8d6e63",
    "KUMA GAWA":     "#ff7043",
    "OYODO GAWA":    "#ffa726",
}

MODEL_COLORS = {
    "STGNN":       "#4fc3f7",
    "No-Graph":    "#66bb6a",
    "Persistence": "#ffa726",
    "Climatology": "#9e9e9e",
    "Observed":    "#ffffff",
}

HORIZON      = 7
MODEL_OPTIONS = ["STGNN", "No-Graph", "Persistence", "Climatology"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.10):
    """Convert #rrggbb to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout():
    """
    Return a Plotly layout dict with NO axis keys and NO margin key.
    Axes and margins are always set via update_xaxes / update_yaxes /
    update_layout after the fact to avoid duplicate-keyword errors.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a1220",
        font=dict(family="DM Sans", color="#c8d0e0", size=12),
        legend=dict(
            bgcolor="rgba(13,31,60,0.85)",
            bordercolor="#1e3a5f",
            borderwidth=1,
        ),
    )


def style_axes(fig):
    """Apply dark grid to every axis in the figure."""
    grid = dict(gridcolor="#1e2d4a", linecolor="#1e2d4a",
                zerolinecolor="#1e2d4a")
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    return fig


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    try:
        with open("Data/processed/config.json") as f:
            config = json.load(f)
        with open("Data/processed/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        so = config["station_order"]
        sc = config["scale_cols"]

        return dict(
            config=config,
            scalers=scalers,
            station_order=so,
            scale_cols=sc,
            pred_stgnn       = np.load("Data/processed/pred_stgnn.npy"),
            pred_nograph     = np.load("Data/processed/pred_nograph.npy"),
            pred_persistence = np.load("Data/processed/pred_persistence.npy"),
            pred_climatology = np.load("Data/processed/pred_climatology.npy"),
            y_true           = np.load("Data/processed/y_true_test.npy"),
            A_norm           = np.load("Data/processed/adj_norm.npy"),
            wide             = pd.read_csv(
                "Data/processed/daily_discharge.csv",
                index_col="date", parse_dates=True,
            ).loc["1993-01-01":"2003-12-31", so],
        )
    except FileNotFoundError as e:
        return {"error": str(e)}


@st.cache_data
def invert_predictions(pred_scaled, s_mins, s_scales,
                       station_order, scale_cols):
    """
    Invert MinMaxScaler then log1p to recover m³/s.
    Arguments are plain numpy arrays so st.cache_data can hash them.
    """
    samples, horizon, N = pred_scaled.shape
    out = np.zeros_like(pred_scaled)
    di  = scale_cols.index("discharge")

    for i in range(N):
        scaled_vals = pred_scaled[:, :, i].flatten()
        # inverse MinMax: x_orig = x_scaled * range + min
        log_vals = scaled_vals * s_scales[i][di] + s_mins[i][di]
        out[:, :, i] = np.expm1(log_vals.reshape(samples, horizon))

    return out


def extract_scaler_arrays(scalers, station_order):
    """Pull min/range arrays out of sklearn scalers for cache-safe hashing."""
    mins   = np.array([scalers[s].data_min_   for s in station_order])
    scales = np.array([scalers[s].data_range_  for s in station_order])
    return mins, scales


@st.cache_data
def build_all_metrics(pred_stgnn, pred_nograph, pred_persist,
                      pred_clim, y_true, station_order):
    all_preds = {
        "STGNN":       pred_stgnn,
        "No-Graph":    pred_nograph,
        "Persistence": pred_persist,
        "Climatology": pred_clim,
    }
    result = {}
    for name, pred in all_preds.items():
        rows = []
        for i, sid in enumerate(station_order):
            p = pred[:, :, i].flatten()
            t = y_true[:, :, i].flatten()
            rows.append({
                "Station": STATIONS[sid]["name"],
                "River":   STATIONS[sid]["river"],
                "MAE":     float(np.mean(np.abs(p - t))),
                "RMSE":    float(np.sqrt(np.mean((p - t) ** 2))),
                "R2":      float(r2_score(t, p)),
            })
        result[name] = pd.DataFrame(rows).set_index("Station")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + PREPARE
# ══════════════════════════════════════════════════════════════════════════════
data = load_all_data()

if "error" in data:
    st.error(f"Could not load data: {data['error']}\n\n"
             "Run notebooks 01–05 first to generate `Data/processed/`.\n"
             "If running on Streamlit Cloud, ensure the `Data/` folder "
             "(with the `processed/` subfolder) is committed to the repo.")
    st.stop()

station_order = data["station_order"]
scale_cols    = data["scale_cols"]
wide          = data["wide"]
A_norm        = data["A_norm"]

s_mins, s_scales = extract_scaler_arrays(data["scalers"], station_order)

y_true_m3s       = invert_predictions(data["y_true"],           s_mins, s_scales, station_order, scale_cols)
pred_stgnn_m3s   = invert_predictions(data["pred_stgnn"],       s_mins, s_scales, station_order, scale_cols)
pred_nograph_m3s = invert_predictions(data["pred_nograph"],     s_mins, s_scales, station_order, scale_cols)
pred_persist_m3s = invert_predictions(data["pred_persistence"], s_mins, s_scales, station_order, scale_cols)
pred_clim_m3s    = invert_predictions(data["pred_climatology"], s_mins, s_scales, station_order, scale_cols)

preds_m3s = {
    "STGNN":       pred_stgnn_m3s,
    "No-Graph":    pred_nograph_m3s,
    "Persistence": pred_persist_m3s,
    "Climatology": pred_clim_m3s,
}

metrics = build_all_metrics(
    pred_stgnn_m3s, pred_nograph_m3s,
    pred_persist_m3s, pred_clim_m3s,
    y_true_m3s, station_order,
)

station_names = [STATIONS[s]["name"] for s in station_order]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 Kyushu River\nDischarge Forecast")
    st.markdown("---")
    page = st.radio(
        "NAVIGATE",
        ["Overview", "Data Explorer", "River Network",
         "Forecast Explorer", "Model Evaluation", "Ablation Study"],
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4a6080; line-height:2.0;'>
    <b style='color:#7a8fa6'>Data</b> · GRDC Kyushu 1993–2003<br>
    <b style='color:#7a8fa6'>Stations</b> · 10 gauging stations<br>
    <b style='color:#7a8fa6'>Task</b> · 7-day ahead forecast<br>
    <b style='color:#7a8fa6'>Model</b> · GRU + GCN (44k params)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <h1 style='font-family:DM Serif Display,serif; font-size:2.6rem;
               color:#e8eaf0; margin-bottom:0;'>
        Kyushu River<br>
        <span style='color:#4fc3f7;'>Discharge Forecasting</span>
    </h1>
    <p style='color:#7a8fa6; font-size:1rem; margin-top:8px;'>
        Spatiotemporal GNN · 10 stations · 7-day horizon · 1993–2003
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # delta_color="inverse" → negative delta shows green (lower MAE = better)
        st.metric("STGNN MAE",     "31.6 m³/s",
                  delta="-25% vs baselines", delta_color="inverse")
    with c2:
        st.metric("STGNN RMSE",    "101.8 m³/s")
    with c3:
        st.metric("STGNN R²",      "0.161")
    with c4:
        st.metric("Train windows", "2,773")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">The Problem</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        River flooding in Kyushu kills people. The region experiences extreme
        monsoon discharge every June–August, with peak flows **20–46× above
        the annual mean** at some stations.

        This project asks: does explicitly encoding **river network topology
        as a graph** improve 7-day discharge forecasts compared to a model
        that treats all stations independently?
        """)

        st.markdown('<div class="section-header" style="margin-top:20px;">Key Findings</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="finding-box">
        🔵 Both learned models achieve <b>~25% lower MAE</b> than persistence
        and climatology baselines across all 10 stations.
        </div>
        <div class="finding-box">
        📈 Error degrades gracefully: <b>25 m³/s at day 1 → 35 m³/s at day 7</b>,
        while persistence collapses to 48 m³/s by day 7.
        </div>
        <div class="warning-box">
        ⚠️ The GCN graph component does <b>not improve over the no-graph
        baseline</b> at daily resolution — upstream–downstream lag = 0 days,
        so the GRU already captures spatial dependencies from the
        multivariate input.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Architecture</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Component": ["Input", "GRU Encoder", "GCN Layer",
                          "Dropout", "Decoder", "Output"],
            "Detail":    ["(B, 14, 10, 10)", "2-layer, hidden=64",
                          "A_norm @ H @ W", "p = 0.1",
                          "Linear → 7 steps", "(B, 7, 10)"],
        }).set_index("Component"), use_container_width=True)

        st.markdown('<div class="section-header" style="margin-top:20px;">Data Split</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Split":   ["Train", "Val", "Test"],
            "Period":  ["1993–2000", "2001", "2002–2003"],
            "Windows": [2773, 316, 674],
        }).set_index("Split"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown('<div class="section-header">Data Explorer</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.multiselect(
            "Select stations",
            options=station_order,
            default=station_order[:3],
            format_func=lambda s: STATIONS[s]["name"],
        )
    with c2:
        log_scale = st.checkbox("Log y-axis", value=False)

    if not selected:
        st.info("Select at least one station.")
        st.stop()

    # ── Time series ───────────────────────────────────────────────────────────
    fig = go.Figure()
    for sid in selected:
        s     = wide[sid].dropna()
        color = RIVER_COLORS[STATIONS[sid]["river"]]
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            name=STATIONS[sid]["name"],
            line=dict(color=color, width=1.2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.08),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} m³/s"
                          "<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        **base_layout(),
        title="Daily Discharge (m³/s)",
        yaxis_title="Discharge (m³/s)",
        yaxis_type="log" if log_scale else "linear",
        xaxis_title="Date",
        height=400,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Statistics</div>',
                unsafe_allow_html=True)
    rows = []
    for sid in selected:
        s = wide[sid].dropna()
        rows.append({
            "Station":      STATIONS[sid]["name"],
            "River":        STATIONS[sid]["river"],
            "Mean (m³/s)":  round(float(s.mean()), 1),
            "Max (m³/s)":   round(float(s.max()), 1),
            "CV":           round(float(s.std() / s.mean()), 2),
            "Max/Mean":     f"{s.max()/s.mean():.1f}×",
            "Missing (%)":  f"{s.isna().mean()*100:.1f}%",
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Station"),
        use_container_width=True,
        column_config={
            "Mean (m³/s)": st.column_config.NumberColumn(format="%.1f"),
            "Max (m³/s)":  st.column_config.NumberColumn(format="%.1f"),
            "CV":          st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # ── Seasonal pattern ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Seasonal Pattern</div>',
                unsafe_allow_html=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig2 = go.Figure()
    for sid in selected:
        s       = wide[sid].dropna()
        monthly = s.groupby(s.index.month).mean()
        fig2.add_trace(go.Bar(
            x=months,
            y=monthly.values.tolist(),
            name=STATIONS[sid]["name"],
            marker_color=RIVER_COLORS[STATIONS[sid]["river"]],
            opacity=0.82,
        ))
    fig2.update_layout(
        **base_layout(),
        title="Monthly Mean Discharge",
        yaxis_title="Mean Q (m³/s)",
        xaxis_title="Month",
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    style_axes(fig2)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RIVER NETWORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "River Network":
    st.markdown('<div class="section-header">River Network Graph</div>',
                unsafe_allow_html=True)

    col_map, col_adj = st.columns([3, 2])

    with col_map:
        st.markdown("**Geographic layout** — node size ∝ catchment area, "
                    "blue lines = flow direction")
        fig = go.Figure()

        # Flow direction lines + midpoint arrow labels
        for up_id, dn_id in TOPOLOGY.items():
            mid_lat = (STATIONS[up_id]["lat"] + STATIONS[dn_id]["lat"]) / 2
            mid_lon = (STATIONS[up_id]["lon"] + STATIONS[dn_id]["lon"]) / 2
            fig.add_trace(go.Scattergeo(
                lat=[STATIONS[up_id]["lat"], STATIONS[dn_id]["lat"]],
                lon=[STATIONS[up_id]["lon"], STATIONS[dn_id]["lon"]],
                mode="lines",
                line=dict(width=2.5, color="#4fc3f7"),
                showlegend=False,
                hoverinfo="none",
            ))
            fig.add_trace(go.Scattergeo(
                lat=[mid_lat],
                lon=[mid_lon],
                mode="text",
                text=["▼"],
                textfont=dict(size=13, color="#4fc3f7"),
                showlegend=False,
                hoverinfo="none",
            ))

        # Nodes grouped by river for legend
        for river_name, color in RIVER_COLORS.items():
            sids = [s for s in station_order
                    if STATIONS[s]["river"] == river_name]
            if not sids:
                continue
            fig.add_trace(go.Scattergeo(
                lat=[STATIONS[s]["lat"] for s in sids],
                lon=[STATIONS[s]["lon"] for s in sids],
                text=[STATIONS[s]["name"] for s in sids],
                customdata=[STATIONS[s]["area"] for s in sids],
                mode="markers+text",
                marker=dict(
                    size=[STATIONS[s]["area"] / 35 for s in sids],
                    color=color,
                    opacity=0.9,
                    line=dict(color="#0a0f1e", width=1.5),
                ),
                textposition="top center",
                textfont=dict(size=9, color="#e8eaf0"),
                name=river_name,
                hovertemplate=(
                    "<b>%{text}</b><br>Area: %{customdata:.0f} km²"
                    f"<br>River: {river_name}<extra></extra>"
                ),
            ))

        fig.update_layout(
            geo=dict(
                scope="asia",
                center=dict(lat=32.5, lon=130.8),
                projection_scale=55,
                showland=True,  landcolor="#0d1f3c",
                showocean=True, oceancolor="#060d1a",
                showcoastlines=True, coastlinecolor="#1e3a5f",
                bgcolor="#0a0f1e",
            ),
            paper_bgcolor="#0a0f1e",
            legend=dict(
                bgcolor="rgba(13,31,60,0.9)",
                bordercolor="#1e3a5f", borderwidth=1,
                font=dict(color="#c8d0e0", size=10),
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_adj:
        st.markdown("**Normalised adjacency matrix** (GCN input)")

        labels = [STATIONS[s]["name"][:8] for s in station_order]
        fig2 = go.Figure(go.Heatmap(
            z=np.round(A_norm, 3).tolist(),
            x=labels,
            y=labels,
            colorscale=[
                [0.0,  "#0a1220"],
                [0.01, "#0d2040"],
                [0.3,  "#1a5080"],
                [1.0,  "#4fc3f7"],
            ],
            zmin=0, zmax=1,
            hovertemplate="From: %{y}<br>To: %{x}<br>"
                          "Weight: %{z:.3f}<extra></extra>",
        ))
        # ── No margin or axis keys in update_layout ───────────────────────
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a1220",
            font=dict(family="DM Sans", color="#c8d0e0", size=12),
            title="A_norm  (D⁻¹/² A D⁻¹/²)",
            height=360,
            margin=dict(l=60, r=10, t=50, b=80),
        )
        fig2.update_xaxes(
            tickangle=45, tickfont=dict(size=8),
            gridcolor="#1e2d4a", linecolor="#1e2d4a",
        )
        fig2.update_yaxes(
            tickfont=dict(size=8),
            gridcolor="#1e2d4a", linecolor="#1e2d4a",
            autorange="reversed",          # station order top → bottom
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Edge table
        st.markdown("**Connected pairs**")
        edge_rows = []
        for up_id, dn_id in TOPOLOGY.items():
            ui = station_order.index(up_id)
            di = station_order.index(dn_id)
            edge_rows.append({
                "Upstream":   STATIONS[up_id]["name"],
                "Downstream": STATIONS[dn_id]["name"],
                "Weight":     round(float(A_norm[ui, di]), 3),
            })
        st.dataframe(
            pd.DataFrame(edge_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Weight": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        st.markdown("""
        <div class="finding-box" style="margin-top:10px;">
        4 isolated nodes (no upstream partner in dataset):<br>
        HINODE-BASHI · MUTABE · TAMANA · ONOBUCHI
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast Explorer":
    st.markdown('<div class="section-header">Forecast Explorer</div>',
                unsafe_allow_html=True)
    st.markdown("Step through test-period windows and inspect 7-day forecasts.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        station_sel = st.selectbox(
            "Station",
            station_order,
            format_func=lambda s: STATIONS[s]["name"],
        )
    with c2:
        models_sel = st.multiselect(
            "Models to overlay",
            options=MODEL_OPTIONS,
            default=["STGNN", "No-Graph"],
        )
    with c3:
        window_idx = st.slider(
            "Window index",
            min_value=0,
            max_value=len(y_true_m3s) - 1,
            value=150,
        )

    sid_idx     = station_order.index(station_sel)
    start_te    = datetime.date(2002, 1, 15)
    window_date = start_te + datetime.timedelta(days=window_idx)
    fcst_dates  = [window_date + datetime.timedelta(days=h)
                   for h in range(HORIZON)]

    # ── 7-day window forecast ─────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fcst_dates,
        y=y_true_m3s[window_idx, :, sid_idx].tolist(),
        name="Observed",
        line=dict(color="#ffffff", width=2.5),
        mode="lines+markers",
        marker=dict(size=8),
    ))
    for model_name in models_sel:
        fig.add_trace(go.Scatter(
            x=fcst_dates,
            y=preds_m3s[model_name][window_idx, :, sid_idx].tolist(),
            name=model_name,
            line=dict(
                color=MODEL_COLORS[model_name],
                width=2,
                dash="solid" if model_name == "STGNN" else "dash",
            ),
            mode="lines+markers",
            marker=dict(size=5),
        ))
    fig.update_layout(
        **base_layout(),
        title=f"{STATIONS[station_sel]['name']} — 7-day forecast "
              f"from {window_date}",
        yaxis_title="Discharge (m³/s)",
        xaxis_title="Date",
        height=380,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Per-window MAE cards
    if models_sel:
        cols = st.columns(len(models_sel))
        for col, mname in zip(cols, models_sel):
            mae = float(np.mean(np.abs(
                preds_m3s[mname][window_idx, :, sid_idx]
                - y_true_m3s[window_idx, :, sid_idx]
            )))
            with col:
                st.metric(f"{mname} MAE", f"{mae:.1f} m³/s")

    # ── Full test period context ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**Full test period — {STATIONS[station_sel]['name']} "
                f"(day-1 forecast)**")

    all_dates = [start_te + datetime.timedelta(days=d)
                 for d in range(len(y_true_m3s))]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=all_dates,
        y=y_true_m3s[:, 0, sid_idx].tolist(),
        name="Observed",
        line=dict(color="#ffffff", width=1.5),
    ))
    for mname, dash in [("STGNN", "solid"), ("Persistence", "dot")]:
        fig2.add_trace(go.Scatter(
            x=all_dates,
            y=preds_m3s[mname][:, 0, sid_idx].tolist(),
            name=mname,
            line=dict(color=MODEL_COLORS[mname], width=1, dash=dash),
            opacity=0.85,
        ))

    # Selected window vertical line — use a Scatter trace to avoid
    # add_vline datetime type errors across Plotly versions
    y_max = float(np.nanmax(y_true_m3s[:, 0, sid_idx])) * 1.15
    fig2.add_trace(go.Scatter(
        x=[window_date, window_date],
        y=[0, y_max],
        mode="lines",
        line=dict(color="#ffa726", width=1.8, dash="dash"),
        name="Selected window",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig2.update_layout(
        **base_layout(),
        height=260,
        yaxis_title="Q (m³/s)",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    style_axes(fig2)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown('<div class="section-header">Model Evaluation</div>',
                unsafe_allow_html=True)

    # ── Aggregated table ──────────────────────────────────────────────────────
    st.markdown("**Aggregated metrics — mean across all 10 stations**")
    summary_rows = []
    for mname, df in metrics.items():
        summary_rows.append({
            "Model":        mname,
            "MAE (m³/s)":  round(float(df["MAE"].mean()), 2),
            "RMSE (m³/s)": round(float(df["RMSE"].mean()), 2),
            "R²":          round(float(df["R2"].mean()),   3),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    st.dataframe(
        summary_df,
        use_container_width=True,
        column_config={
            "MAE (m³/s)":  st.column_config.NumberColumn(format="%.2f"),
            "RMSE (m³/s)": st.column_config.NumberColumn(format="%.2f"),
            "R²":          st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(
        ["📊 MAE by Station", "📈 R² by Station", "⏱ Error by Horizon"]
    )

    with tab1:
        fig = go.Figure()
        for mname, df in metrics.items():
            fig.add_trace(go.Bar(
                name=mname,
                x=station_names,
                y=[round(float(v), 2) for v in df["MAE"].values],
                marker_color=MODEL_COLORS[mname],
                opacity=0.85,
            ))
        fig.update_layout(
            **base_layout(),
            barmode="group",
            yaxis_title="MAE (m³/s)",
            title="Mean Absolute Error by Station and Model",
            height=420,
            margin=dict(l=10, r=10, t=50, b=80),
        )
        fig.update_xaxes(tickangle=-30, gridcolor="#1e2d4a",
                         linecolor="#1e2d4a")
        fig.update_yaxes(gridcolor="#1e2d4a", linecolor="#1e2d4a")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        for mname, df in metrics.items():
            fig.add_trace(go.Bar(
                name=mname,
                x=station_names,
                y=[round(float(v), 3) for v in df["R2"].values],
                marker_color=MODEL_COLORS[mname],
                opacity=0.85,
            ))
        fig.add_hline(y=0, line_color="#ef5350",
                      line_width=1.2, line_dash="dash")
        fig.update_layout(
            **base_layout(),
            barmode="group",
            yaxis_title="R²",
            title="R² Score by Station and Model",
            height=420,
            margin=dict(l=10, r=10, t=50, b=80),
        )
        fig.update_xaxes(tickangle=-30, gridcolor="#1e2d4a",
                         linecolor="#1e2d4a")
        fig.update_yaxes(gridcolor="#1e2d4a", linecolor="#1e2d4a")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["MAE vs Forecast Horizon",
                            "RMSE vs Forecast Horizon"],
        )
        xs = list(range(1, HORIZON + 1))
        for mname, pred in preds_m3s.items():
            color  = MODEL_COLORS[mname]
            mae_h  = [round(float(np.mean(np.abs(
                          pred[:, h, :] - y_true_m3s[:, h, :]))), 2)
                      for h in range(HORIZON)]
            rmse_h = [round(float(np.sqrt(np.mean(
                          (pred[:, h, :] - y_true_m3s[:, h, :]) ** 2))), 2)
                      for h in range(HORIZON)]
            fig.add_trace(
                go.Scatter(x=xs, y=mae_h, name=mname,
                           mode="lines+markers",
                           line=dict(color=color, width=2),
                           marker=dict(size=6),
                           showlegend=True),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=xs, y=rmse_h, name=mname,
                           mode="lines+markers",
                           line=dict(color=color, width=2),
                           marker=dict(size=6),
                           showlegend=False),
                row=1, col=2,
            )
        fig.update_layout(
            **base_layout(),
            height=380,
            margin=dict(l=10, r=10, t=50, b=40),
        )
        fig.update_xaxes(tickvals=xs, title_text="Days ahead",
                         gridcolor="#1e2d4a", linecolor="#1e2d4a")
        fig.update_yaxes(gridcolor="#1e2d4a", linecolor="#1e2d4a")
        fig.update_yaxes(title_text="MAE (m³/s)",  row=1, col=1)
        fig.update_yaxes(title_text="RMSE (m³/s)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-station detail ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Per-station detail — STGNN vs No-Graph**")
    detail = pd.DataFrame({
        "STGNN MAE":  [round(float(v), 2) for v in metrics["STGNN"]["MAE"].values],
        "NG MAE":     [round(float(v), 2) for v in metrics["No-Graph"]["MAE"].values],
        "STGNN RMSE": [round(float(v), 2) for v in metrics["STGNN"]["RMSE"].values],
        "NG RMSE":    [round(float(v), 2) for v in metrics["No-Graph"]["RMSE"].values],
        "STGNN R²":   [round(float(v), 3) for v in metrics["STGNN"]["R2"].values],
        "NG R²":      [round(float(v), 3) for v in metrics["No-Graph"]["R2"].values],
    }, index=station_names)
    st.dataframe(
        detail,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(
                format="%.3f" if "R²" in col else "%.2f"
            )
            for col in detail.columns
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ablation Study":
    st.markdown('<div class="section-header">Ablation Study</div>',
                unsafe_allow_html=True)
    st.markdown("Does the river graph topology actually help? "
                "Here is the honest answer.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("STGNN MAE",   "31.61 m³/s")
        st.metric("STGNN RMSE",  "101.78 m³/s")
        st.metric("STGNN R²",    "0.161")
    with c2:
        # MAE higher for No-Graph → bad → delta_color="inverse" makes it red
        st.metric("No-Graph MAE",  "32.03 m³/s",
                  delta="+0.42 vs STGNN",  delta_color="inverse")
        # RMSE lower for No-Graph → good → delta_color="inverse" makes it green
        st.metric("No-Graph RMSE", "101.12 m³/s",
                  delta="-0.66 vs STGNN",  delta_color="inverse")
        # R² higher for No-Graph → good → delta_color="normal" makes it green
        st.metric("No-Graph R²",   "0.169",
                  delta="+0.008 vs STGNN", delta_color="normal")
    with c3:
        st.markdown("""
        <div class="finding-box" style="padding:22px; font-size:1rem;
             margin-top:4px;">
        The models are <b>statistically tied</b>.<br><br>
        No metric consistently favours STGNN across all 10 stations.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Per-station MAE: STGNN vs No-Graph**")

    stgnn_mae   = [round(float(v), 2) for v in metrics["STGNN"]["MAE"].values]
    nograph_mae = [round(float(v), 2) for v in metrics["No-Graph"]["MAE"].values]
    scolors     = [RIVER_COLORS[STATIONS[s]["river"]] for s in station_order]

    # Compute range from actual data
    all_vals = stgnn_mae + nograph_mae
    ax_max   = round(max(all_vals) * 1.18, 0)
    ax_min   = 0.0

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[ax_min, ax_max],
        y=[ax_min, ax_max],
        mode="lines",
        line=dict(color="#2a3a5a", dash="dash", width=1.5),
        name="Equal performance",
        showlegend=True,
        hoverinfo="skip",
    ))

    # One point per station — use individual traces so river colours show
    seen_rivers = set()
    for i, sid in enumerate(station_order):
        river     = STATIONS[sid]["river"]
        show_leg  = river not in seen_rivers
        seen_rivers.add(river)
        fig.add_trace(go.Scatter(
            x=[stgnn_mae[i]],
            y=[nograph_mae[i]],
            mode="markers+text",
            text=[STATIONS[sid]["name"]],
            textposition="top center",
            textfont=dict(size=9, color="#c8d0e0"),
            marker=dict(
                size=14,
                color=scolors[i],
                line=dict(color="#0a0f1e", width=1.5),
            ),
            name=river,
            legendgroup=river,
            showlegend=show_leg,
            hovertemplate=(
                f"<b>{STATIONS[sid]['name']}</b><br>"
                f"STGNN MAE: {stgnn_mae[i]:.1f} m³/s<br>"
                f"No-Graph MAE: {nograph_mae[i]:.1f} m³/s"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **base_layout(),
        height=460,
        margin=dict(l=60, r=20, t=60, b=60),
        title="Points above diagonal → No-Graph wins | Below → STGNN wins",
        xaxis=dict(
            title="STGNN MAE (m³/s)",
            range=[ax_min, ax_max],
            gridcolor="#1e2d4a",
            linecolor="#1e2d4a",
        ),
        yaxis=dict(
            title="No-Graph MAE (m³/s)",
            range=[ax_min, ax_max],
            gridcolor="#1e2d4a",
            linecolor="#1e2d4a",
        ),
        annotations=[
            dict(
                x=ax_max * 0.60, y=ax_max * 0.50,
                text="← STGNN better",
                showarrow=False,
                font=dict(color="#4fc3f7", size=11),
            ),
            dict(
                x=ax_max * 0.35, y=ax_max * 0.68,
                text="No-Graph better →",
                showarrow=False,
                font=dict(color="#66bb6a", size=11),
            ),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Why the graph did not help**")

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("""
        <div class="finding-box">
        <b>🕐 Lag = 0 days</b><br><br>
        Peak upstream–downstream correlation occurs at zero lag for all
        three river pairs. At daily resolution the GRU already observes
        all 10 stations simultaneously and learns these relationships
        without graph message passing.
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="finding-box">
        <b>🕸️ Sparse graph</b><br><br>
        Only 3 edges out of 90 possible. The adjacency matrix is 97% zeros.
        The GCN layer has almost no structural information to propagate
        and degenerates to a simple linear projection.
        </div>
        """, unsafe_allow_html=True)
    with cc:
        st.markdown("""
        <div class="finding-box">
        <b>📊 Dataset size</b><br><br>
        2,773 training windows is enough for the GRU but may be
        insufficient for the GCN to learn marginal spatial contributions
        beyond the rich multivariate temporal features.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box" style="margin-top:16px;">
    <b>When graph topology would help:</b> hourly data (lag &gt; 0),
    denser gauge networks, or larger basins with spatially heterogeneous
    rainfall. This project precisely defines those boundary conditions —
    which is itself a publishable negative result.
    </div>
    """, unsafe_allow_html=True)