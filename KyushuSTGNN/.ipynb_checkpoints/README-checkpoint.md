# Kyushu River Discharge Forecasting with Spatiotemporal GNN

A machine learning engineering project that predicts daily river discharge
at 10 gauging stations across Kyushu, Japan, 7 days ahead, using a
Spatiotemporal Graph Neural Network (STGNN) built on real GRDC hydrological data.

---

## The Problem

River flooding in Kyushu kills people. The region experiences extreme monsoon
discharge every June–August, with peak flows 20–46× above the annual mean at
some stations. Accurate multi-day forecasts give emergency managers time to act.

Existing approaches either model each gauging station independently — ignoring
the fact that what happens upstream today directly affects downstream conditions
tomorrow — or rely on simple statistical baselines that cannot capture the
nonlinear, seasonal dynamics of monsoon-driven discharge.

**Core question:** Does explicitly encoding river network topology as a graph
improve 7-day discharge forecasts compared to a model that treats all stations
independently?

---

## Data

- **Source:** Global Runoff Data Centre (GRDC), Kyushu subregion, Japan
- **Stations:** 10 gauging stations across 7 rivers (Chikugo, Onga, Matsuura,
  Kikuchi, Sendai, Kuma, Oyodo)
- **Period:** 1993–2003 (daily resolution, ~4,017 days)
- **Format:** GRDC standard semicolon-delimited `.txt` files

| Station | River | Catchment (km²) |
|---|---|---|
| SENOSHITA | CHIKUGO GAWA | 2,315 |
| ARASE | CHIKUGO GAWA | 1,443 |
| HINODE-BASHI | ONGA GAWA | 695 |
| MUTABE | MATSUURA GAWA | 275 |
| TAMANA | KIKUCHI GAWA | 906 |
| ONOBUCHI | SENDAI GAWA | 1,348 |
| YOKOISHI | KUMA GAWA | 1,856 |
| HITOYOSHI | KUMA GAWA | 1,137 |
| KASHIWADA | OYODO GAWA | 2,126 |
| HIWATASHI | OYODO GAWA | 861 |

---

## Approach

### Graph construction
The river network is encoded as a directed graph where edges connect upstream
to downstream stations on the same river. Three pairs are connected
(Chikugo, Kuma, Oyodo). Edge weights are the upstream/downstream catchment
area ratio. The adjacency matrix is degree-normalised following
Kipf & Welling (2017).

### Feature engineering
Each station receives 10 features per day:

| Feature | Type | Description |
|---|---|---|
| discharge | dynamic | log(1+Q) at time t |
| lag_1d / 3d / 7d / 14d | dynamic | Autoregressive lags |
| roll_7d_mean / std | dynamic | 7-day rolling statistics |
| sin_doy / cos_doy | seasonal | Day-of-year encoding |
| area_norm | static | Log-normalised catchment area |

### Model architecture
A three-stage STGNN operating on tensors of shape
`(batch, 14 days, 10 stations, 10 features)`:
Input (B,14,10,10)
→ GRU Encoder       per-node 2-layer GRU, hidden=64
→ GCN Layer         A_norm @ H @ W, ReLU, dim=64
→ Dropout (p=0.1)
→ Linear Decoder    64 → 7 steps per node
Output (B,7,10)

**44,167 trainable parameters.** Trained with Adam (lr=1e-3),
ReduceLROnPlateau scheduler, early stopping (patience=15), gradient
clipping (max_norm=1.0). Best model selected on validation MSE.

### Ablation
An identical model with `use_graph=False` disables graph message passing
while keeping all other components fixed. This isolates the contribution
of the river topology.

### Baselines
- **Persistence:** last observed discharge repeated for 7 days
- **Climatology:** per-day-of-year training mean for each forecast step

### Train / val / test split

| Split | Period | Windows |
|---|---|---|
| Train | 1993–2000 | 2,773 |
| Val | 2001 | 316 |
| Test | 2002–2003 | 674 |

---

## Results

### Aggregated metrics (test set, 2002–2003)

| Model | MAE (m³/s) | RMSE (m³/s) | R² |
|---|---|---|---|
| **STGNN** | **31.61** | 101.78 | 0.161 |
| No-Graph | 32.03 | **101.12** | **0.169** |
| Persistence | 42.15 | 125.88 | -0.308 |
| Climatology | 42.15 | 110.92 | 0.011 |

Both learned models achieve **~25% lower MAE** than either baseline.
The horizon error plot shows graceful degradation from ~25 m³/s at day 1
to ~35 m³/s at day 7, while persistence collapses to ~48 m³/s by day 7.

### Ablation result

The no-graph model matches STGNN within measurement noise across all metrics
and all 10 stations. This is the central scientific finding of the project.

**Why the graph did not help:**

1. **Lag = 0 days.** Cross-correlation analysis showed peak upstream-downstream
   correlation at zero lag for all three river pairs. At daily resolution,
   flood signals arrive at downstream gauges within the same calendar day.
   The GRU already observes all 10 stations simultaneously as input features
   and can learn this relationship without explicit message passing.

2. **Sparse graph.** Only 3 edges out of 90 possible connections exist.
   The GCN layer operates on a 97% zero adjacency matrix, providing
   almost no structural information to propagate.

3. **Dataset size.** 2,773 training windows is sufficient for the GRU to
   learn temporal patterns but may not provide enough signal for the GCN
   to learn marginal spatial contributions beyond the rich temporal features.

**When graph topology would help:** hourly data (where upstream-downstream
lag > 0), denser gauge networks, or larger basins with spatially heterogeneous
rainfall forcing.

---

## Project Structure
kyushu-stgnn/
│
├── data/
│   ├── raw/                    # GRDC .txt files + GeoJSON
│   └── processed/              # Generated outputs (numpy, pkl, json)
│
├── 01_eda.ipynb                # Discharge distributions, seasonality, missingness
├── 02_graph.ipynb              # Adjacency matrix construction, river network map
├── 03_dataset.ipynb            # Feature engineering, sliding windows, splits
├── 04_model.ipynb              # STGNN architecture, forward pass verification
├── 05_train.ipynb              # Training loop, loss curves, baseline predictions
├── 06_evaluate.ipynb           # MAE/RMSE/R², ablation table, forecast plots
│
├── requirements.txt
└── README.md

---

## Reproducing the Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place GRDC data
Copy your GRDC `.txt` files and GeoJSON into `data/raw/`:

data/raw/
├── 2590100_Q_Day.Cmd.txt
├── 2590100_Q_Month.txt
├── ... (all 20 station files)
├── stationbasins.geojson
└── subregions.geojson

### 3. Run notebooks in order

01_eda.ipynb       →  generates data/processed/daily_discharge.csv
02_graph.ipynb     →  generates data/processed/adj_norm.npy + station_order.json
03_dataset.ipynb   →  generates X/y train/val/test tensors + scalers.pkl
04_model.ipynb     →  defines architecture, verifies forward pass
05_train.ipynb     →  trains both models, generates predictions
06_evaluate.ipynb  →  computes all metrics and plots

---

## Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
networkx
torch

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Temporal encoder | GRU (not LSTM) | Fewer parameters, comparable performance |
| Shared GRU weights | Yes, across all nodes | Forces generalisation, reduces parameters |
| Adjacency | Topology-based, not correlation | Correlation is shared monsoon signal, not causality |
| Target transform | log(1+Q) then MinMaxScale | Compresses heavy tail, stabilises training |
| Scaler fit | Train set only | Prevents data leakage into val/test |
| Window drop | Any NaN → drop entire window | Clean training signal, negligible data loss |
| Early stopping | Patience=15 on val MSE | Prevents overfitting, allows LR schedule to act |

---

## Limitations and Future Work

- **Flood peak underestimation.** MSE loss penalises all errors equally.
  A weighted or quantile loss function would improve extreme event prediction.
- **No meteorological forcing.** Rainfall, temperature, and soil moisture
  data would likely improve forecasts significantly, especially for lead times
  beyond 3 days where antecedent discharge is less informative.
- **Daily resolution limits graph utility.** Resampling to hourly or
  sub-daily intervals would introduce non-zero upstream-downstream lags
  and is the most promising next step for making the graph component useful.
- **Static graph.** The adjacency matrix is fixed. A learned or dynamic
  adjacency could discover non-topological spatial dependencies.

---