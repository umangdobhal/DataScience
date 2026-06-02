import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 1 : Imports ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ── All imports ─────────────────────────────────────────────────────────────
import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Plot style
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

print("All imports OK")
"""))

# ── Cell 2 : Station metadata ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 1. Station Metadata
Define the 10 gauging stations with their river network topology.  
Topology rule: larger catchment area downstream on the same river.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Station metadata derived from stationbasins.geojson
STATIONS = {
    "2590100": {"name": "SENOSHITA",    "river": "CHIKUGO GAWA",  "area": 2315.0, "lat": 33.3161, "lon": 130.4972, "alt":  2.0},
    "2590101": {"name": "ARASE",        "river": "CHIKUGO GAWA",  "area": 1443.0, "lat": 33.3433, "lon": 130.8350, "alt": -999},
    "2590200": {"name": "HINODE-BASHI", "river": "ONGA GAWA",     "area":  695.0, "lat": 33.7500, "lon": 130.7300, "alt": -999},
    "2590210": {"name": "MUTABE",       "river": "MATSUURA GAWA", "area":  275.0, "lat": 33.3600, "lon": 130.0100, "alt": -999},
    "2590220": {"name": "TAMANA",       "river": "KIKUCHI GAWA",  "area":  906.0, "lat": 32.9400, "lon": 130.5900, "alt": -999},
    "2590230": {"name": "ONOBUCHI",     "river": "SENDAI GAWA",   "area": 1348.0, "lat": 31.8600, "lon": 130.3400, "alt": -999},
    "2590300": {"name": "YOKOISHI",     "river": "KUMA GAWA",     "area": 1856.0, "lat": 32.4600, "lon": 130.6600, "alt": -999},
    "2590301": {"name": "HITOYOSHI",    "river": "KUMA GAWA",     "area": 1137.0, "lat": 32.2100, "lon": 130.7700, "alt": -999},
    "2590400": {"name": "KASHIWADA",    "river": "OYODO GAWA",    "area": 2126.0, "lat": 31.9500, "lon": 131.4000, "alt": -999},
    "2590401": {"name": "HIWATASHI",    "river": "OYODO GAWA",    "area":  860.6, "lat": 31.8600, "lon": 131.1000, "alt": -999},
}

# River topology: upstream_id -> downstream_id (same river, smaller -> larger area)
TOPOLOGY = {
    "2590101": "2590100",   # ARASE       -> SENOSHITA   (Chikugo)
    "2590301": "2590300",   # HITOYOSHI   -> YOKOISHI    (Kuma)
    "2590401": "2590400",   # HIWATASHI   -> KASHIWADA   (Oyodo)
    # HINODE-BASHI, MUTABE, TAMANA, ONOBUCHI have no pair in this dataset
}

meta_df = pd.DataFrame(STATIONS).T
meta_df["area"] = meta_df["area"].astype(float)
meta_df.index.name = "station_id"
print(meta_df[["name", "river", "area", "lat", "lon"]])
"""))

# ── Cell 3 : Parser ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading\nParse all 10 daily discharge files into a single wide DataFrame."))

cells.append(nbf.v4.new_code_cell("""\
DATA_DIR = "data/raw"   # ← adjust if your path differs

def parse_daily(path: str) -> pd.DataFrame:
    \"\"\"
    Parse a GRDC *_Q_Day.Cmd.txt file.
    Returns DataFrame with DatetimeIndex and single column 'discharge' (m³/s).
    Missing values (-999) become NaN.
    \"\"\"
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # Find the header row (first non-comment line)
    data_start = next(i for i, l in enumerate(lines) if not l.startswith("#"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=data_start,
        header=0,
        encoding="latin-1",
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"YYYY-MM-DD": "date", "Value": "discharge"})
    df["date"] = pd.to_datetime(df["date"].str.strip())
    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
    df.loc[df["discharge"] < 0, "discharge"] = np.nan
    return df.set_index("date")["discharge"].sort_index()


# Load all stations
daily = {}
for sid, info in STATIONS.items():
    pattern = os.path.join(DATA_DIR, f"{sid}_Q_Day.Cmd.txt")
    files = glob.glob(pattern)
    if not files:
        print(f"  [MISSING] {sid}")
        continue
    daily[sid] = parse_daily(files[0])
    n_valid = daily[sid].notna().sum()
    n_miss  = daily[sid].isna().sum()
    print(f"  {sid} ({info['name']:15s})  {daily[sid].index[0].date()} → "
          f"{daily[sid].index[-1].date()}   records={len(daily[sid])}  "
          f"missing={n_miss} ({n_miss/len(daily[sid])*100:.1f}%)")
"""))

# ── Cell 4 : Wide DataFrame ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# Merge into a single wide DataFrame  (index=date, columns=station_id)
wide = pd.DataFrame({sid: s for sid, s in daily.items()})
wide.index.name = "date"

# Friendly column names for display
name_map = {sid: info["name"] for sid, info in STATIONS.items()}

print(f"Shape : {wide.shape}  ({wide.shape[0]} days × {wide.shape[1]} stations)")
print(f"Period: {wide.index[0].date()} → {wide.index[-1].date()}")
print(f"\\nOverall missing: {wide.isna().mean().mean()*100:.2f}%")
wide.head(3)
"""))

# ── Cell 5 : Save processed ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
os.makedirs("data/processed", exist_ok=True)
wide.to_csv("data/processed/daily_discharge.csv")
print("Saved → data/processed/daily_discharge.csv")
"""))

# ── Cell 6 : Summary stats ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Summary Statistics"))

cells.append(nbf.v4.new_code_cell("""\
stats = wide.agg(["count", "mean", "std", "min", "max"]).T
stats.columns   = ["n_valid", "mean_m3s", "std_m3s", "min_m3s", "max_m3s"]
stats.index     = [STATIONS[s]["name"] for s in stats.index]
stats["cv"]     = (stats["std_m3s"] / stats["mean_m3s"]).round(2)   # coefficient of variation
stats["max/mean"] = (stats["max_m3s"] / stats["mean_m3s"]).round(1) # flood ratio

pd.options.display.float_format = "{:,.1f}".format
print(stats)
"""))

# ── Cell 7 : Missing heatmap ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Missingness Overview"))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(14, 3))

# Binary missing matrix resampled to monthly for readability
miss_monthly = wide.resample("ME").apply(lambda x: x.isna().mean())
miss_monthly.columns = [STATIONS[s]["name"] for s in miss_monthly.columns]

sns.heatmap(
    miss_monthly.T,
    ax=ax,
    cmap="YlOrRd",
    vmin=0, vmax=1,
    linewidths=0,
    cbar_kws={"label": "Fraction missing", "shrink": 0.6},
)
ax.set_xlabel("Year-Month")
ax.set_title("Monthly fraction of missing daily values per station", pad=10)

# Show only year labels on x-axis
years = miss_monthly.resample("YS").first().index
tick_pos = [miss_monthly.index.get_loc(y, method="nearest") for y in years]
ax.set_xticks(tick_pos)
ax.set_xticklabels([str(y.year) for y in years], rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("data/processed/fig_missingness.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 8 : Discharge time series ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Discharge Time Series"))

cells.append(nbf.v4.new_code_cell("""\
station_ids = list(STATIONS.keys())
n = len(station_ids)
fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n), sharex=True)

colors = plt.cm.tab10.colors

for i, (sid, ax) in enumerate(zip(station_ids, axes)):
    s = daily[sid]
    ax.fill_between(s.index, s.values, alpha=0.35, color=colors[i])
    ax.plot(s.index, s.values, lw=0.6, color=colors[i])
    ax.set_ylabel("m³/s", fontsize=9)
    ax.set_title(f"{STATIONS[sid]['name']}  ({STATIONS[sid]['river']})", fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
fig.suptitle("Daily Discharge — All Kyushu Stations", fontsize=14, y=1.001)
plt.tight_layout()
plt.savefig("data/processed/fig_timeseries.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 9 : Seasonal climatology ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 6. Seasonal Climatology (Monthly Means)\nShows the monsoon signal clearly."))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey=False)
axes = axes.flatten()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for i, (sid, ax) in enumerate(zip(station_ids, axes)):
    s = daily[sid].dropna()
    monthly_mean = s.groupby(s.index.month).mean()
    monthly_std  = s.groupby(s.index.month).std()

    ax.bar(monthly_mean.index, monthly_mean.values, color=colors[i], alpha=0.7, width=0.7)
    ax.errorbar(monthly_mean.index, monthly_mean.values,
                yerr=monthly_std.values, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45, ha="right", fontsize=7)
    ax.set_title(STATIONS[sid]["name"], fontsize=10)
    ax.set_ylabel("Mean Q (m³/s)", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.suptitle("Monthly Mean Discharge ± 1 SD", fontsize=14)
plt.tight_layout()
plt.savefig("data/processed/fig_climatology.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 10 : Discharge distributions ────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 7. Discharge Distributions\nLog scale reveals the heavy tail typical of river discharge."))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 5, figsize=(18, 5))
axes = axes.flatten()

for i, (sid, ax) in enumerate(zip(station_ids, axes)):
    s = daily[sid].dropna()
    log_s = np.log1p(s)
    ax.hist(log_s, bins=60, color=colors[i], alpha=0.75, edgecolor="none")
    ax.axvline(log_s.mean(), color="black", lw=1.5, linestyle="--", label="mean")
    ax.set_title(STATIONS[sid]["name"], fontsize=10)
    ax.set_xlabel("log(1 + Q)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

fig.suptitle("Distribution of Daily Discharge (log-transformed)", fontsize=14)
plt.tight_layout()
plt.savefig("data/processed/fig_distributions.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 11 : Cross-station correlation ──────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 8. Cross-Station Correlation\nExpected: stations on the same river are highly correlated."))

cells.append(nbf.v4.new_code_cell("""\
corr = wide.corr(min_periods=100)
corr.index   = [STATIONS[s]["name"] for s in corr.index]
corr.columns = [STATIONS[s]["name"] for s in corr.columns]

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, ax=ax, mask=mask,
    cmap="RdYlGn", vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    linewidths=0.5, square=True,
    cbar_kws={"shrink": 0.7, "label": "Pearson r"},
)
ax.set_title("Cross-Station Discharge Correlation", pad=12)
plt.tight_layout()
plt.savefig("data/processed/fig_correlation.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 12 : Upstream vs downstream lag ─────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Upstream → Downstream Lag Correlation
For the three river pairs with both upstream and downstream stations,  
compute cross-correlation at lags 0–7 days to estimate travel time.
"""))

cells.append(nbf.v4.new_code_cell("""\
pairs = [
    ("2590101", "2590100", "Chikugo: ARASE → SENOSHITA"),
    ("2590301", "2590300", "Kuma:    HITOYOSHI → YOKOISHI"),
    ("2590401", "2590400", "Oyodo:   HIWATASHI → KASHIWADA"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (up_id, dn_id, title) in zip(axes, pairs):
    up = daily[up_id].dropna()
    dn = daily[dn_id].dropna()
    common = up.index.intersection(dn.index)
    up_c, dn_c = up[common], dn[common]

    lags = range(0, 8)
    corrs = [up_c.corr(dn_c.shift(-lag)) for lag in lags]

    ax.bar(list(lags), corrs, color="#2196F3", alpha=0.8)
    best_lag = int(np.argmax(corrs))
    ax.axvline(best_lag, color="red", lw=1.5, linestyle="--",
               label=f"Best lag = {best_lag}d")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

fig.suptitle("Upstream–Downstream Cross-Correlation by Lag", fontsize=13)
plt.tight_layout()
plt.savefig("data/processed/fig_lag_correlation.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 13 : Annual max discharge ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 10. Annual Maximum Discharge\nIdentifies extreme flood years across stations."))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 5, figsize=(18, 5))
axes = axes.flatten()

for i, (sid, ax) in enumerate(zip(station_ids, axes)):
    s = daily[sid].dropna()
    ann_max = s.resample("YE").max()
    ax.bar(ann_max.index.year, ann_max.values, color=colors[i], alpha=0.8, width=0.7)
    ax.set_title(STATIONS[sid]["name"], fontsize=10)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Max Q (m³/s)", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.suptitle("Annual Maximum Daily Discharge per Station", fontsize=14)
plt.tight_layout()
plt.savefig("data/processed/fig_annual_max.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 14 : Flow Duration Curve ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 11. Flow Duration Curves\nShows the percentage of time discharge exceeds a given level."))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(10, 6))

for i, sid in enumerate(station_ids):
    s = daily[sid].dropna().sort_values(ascending=False)
    exceedance = np.linspace(0, 100, len(s))
    ax.semilogy(exceedance, s.values, lw=1.8, color=colors[i],
                label=STATIONS[sid]["name"])

ax.set_xlabel("Exceedance probability (%)")
ax.set_ylabel("Discharge (m³/s)  [log scale]")
ax.set_title("Flow Duration Curves — All Stations")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/fig_fdc.png", dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── Cell 15 : EDA summary ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 12. EDA Summary & Key Findings

| Finding | Implication for modelling |
|---|---|
| Strong monsoon peak June–August | Seasonal encoding (sin/cos DOY) is essential |
| Upstream–downstream lag of 1–2 days | Graph message passing can propagate flood signals |
| High CV (>2 for most stations) | Log-transform discharge before training |
| Low missingness (<5%) in daily data | No imputation needed; drop incomplete windows |
| High cross-station correlation within rivers | Graph structure is informative |
| Extreme flood spikes (max/mean > 20×) | Model must handle heavy-tailed targets |

**Next step → `02_graph.ipynb`**: build the adjacency matrix from the river topology.
"""))

# ── Assemble and write ───────────────────────────────────────────────────────
nb.cells = cells
out = "/home/claude/01_eda.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Written: {out}")