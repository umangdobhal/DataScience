# Kyushu River Discharge Forecasting with Spatiotemporal GNN

A machine learning engineering project that predicts daily river discharge
at 10 gauging stations across Kyushu, Japan, 7 days ahead, using a
Spatiotemporal Graph Neural Network (STGNN) built on real GRDC hydrological
data — with an interactive Streamlit dashboard for exploration.

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
- **Period:** 1993–2003 (daily resolution, ~4,017 days per station)
- **Format:** GRDC standard semicolon-delimited `.txt` files
- **Missingness:** Effectively 0% after clipping to 1993 (TAMANA: 0.4%)

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
Kipf & Welling (2017). Four stations (HINODE-BASHI, MUTABE, TAMANA, ONOBUCHI)
have no topological pair in this dataset and act as independent nodes.

### Feature engineering
Each station receives 10 features per day, all computed on log-transformed
discharge:

| Feature | Type | Description |
|---|---|---|
| discharge | dynamic | log(1+Q) at time t |
| lag_1d / 3d / 7d / 14d | dynamic | Autoregressive lags |
| roll_7d_mean / std | dynamic | 7-day rolling statistics (shift=1, no leakage) |
| sin_doy / cos_doy | seasonal | Day-of-year encoding for monsoon seasonality |
| area_norm | static | Log-normalised catchment area per node |

### Model architecture
A three-stage STGNN operating on tensors of shape
`(batch, 14 days, 10 stations, 10 features)`:

Input  (B, 14, 10, 10)
→ GRU Encoder      2-layer GRU, hidden=64, shared weights across nodes
→ GCN Layer        A_norm @ H @ W, ReLU, dim=64
→ Dropout          p=0.1
→ Linear Decoder   64 → 7 steps per node
Output (B, 7, 10)

**44,167 trainable parameters.** The GRU dominates (89% of parameters);
the GCN is deliberately lightweight since the graph topology is fixed.

### Training
- **Optimiser:** Adam, lr=1e-3, weight_decay=1e-4
- **Scheduler:** ReduceLROnPlateau × 0.5, patience=7 epochs
- **Early stopping:** patience=15 on validation MSE
- **Gradient clipping:** max_norm=1.0
- **Loss:** Masked MSE on scaled log(1+Q)
- **Seed:** 42 for reproducibility
- STGNN converged at epoch 61, best val MSE = 0.0095
- No-Graph converged at epoch 58, best val MSE = 0.0094

### Ablation
An identical model with `use_graph=False` disables graph message passing
while keeping all other components fixed — same parameter count, same
training procedure. This isolates the contribution of the river topology.

### Baselines
- **Persistence:** last observed discharge repeated for all 7 horizon steps
- **Climatology:** per-day-of-year training mean for each forecast step and station

### Train / val / test split

| Split | Period | Windows |
|---|---|---|
| Train | 1993–2000 | 2,773 |
| Val | 2001 | 316 |
| Test | 2002–2003 | 674 |

Input window = 14 days, forecast horizon = 7 days, sliding step = 1 day.
Windows containing any NaN are dropped entirely.

---

## Results

### Aggregated metrics (test set, 2002–2003)

| Model | MAE (m³/s) | RMSE (m³/s) | R² |
|---|---|---|---|
| **STGNN** | **31.61** | 101.78 | 0.161 |
| No-Graph | 32.03 | **101.12** | **0.169** |
| Persistence | 42.15 | 125.88 | −0.308 |
| Climatology | 42.15 | 110.92 | 0.011 |

Both learned models achieve **~25% lower MAE** than either baseline.
Error degrades gracefully from ~25 m³/s at day 1 to ~35 m³/s at day 7,
while persistence collapses to ~48 m³/s by day 7.

### Per-station highlights
- **KASHIWADA** is the hardest station (RMSE 224 m³/s, largest catchment,
  highest flood peaks up to 3,798 m³/s in the test period)
- **MUTABE** is the easiest (RMSE 20 m³/s, smallest catchment, 275 km²)
- R² ranges from 0.09 (MUTABE) to 0.27 (SENOSHITA) — modest but meaningful
  for a 7-day multi-site forecast on a heavy-tailed target

### Ablation result

The no-graph model matches STGNN within measurement noise across all metrics
and all 10 stations. This is the central scientific finding of the project.

**Why the graph did not help:**

1. **Lag = 0 days.** EDA cross-correlation showed peak upstream–downstream
   correlation at zero lag for all three river pairs. At daily resolution,
   flood signals arrive at downstream gauges within the same calendar day.
   The GRU already observes all 10 stations simultaneously and can learn
   this relationship without explicit message passing.

2. **Sparse graph.** Only 3 edges out of 90 possible connections exist.
   The GCN layer operates on a 97% zero adjacency matrix, providing
   almost no structural information to propagate — it degenerates to a
   linear projection.

3. **Dataset size.** 2,773 training windows is sufficient for the GRU to
   learn temporal patterns but may not provide enough signal for the GCN
   to learn marginal spatial contributions beyond the rich temporal features.

**When graph topology would help:** hourly data (lag > 0), denser gauge
networks, or larger basins with spatially heterogeneous rainfall forcing.

---

## Project Structure
kyushu-stgnn/
│
├── data/
│   ├── raw/                    # GRDC .txt files + GeoJSON (not tracked by git)
│   └── processed/              # Generated outputs (numpy, pkl, json)
│       ├── daily_discharge.csv
│       ├── adj_norm.npy
│       ├── station_order.json
│       ├── X_train.npy / y_train.npy
│       ├── X_val.npy   / y_val.npy
│       ├── X_test.npy  / y_test.npy
│       ├── scalers.pkl
│       ├── config.json
│       ├── best_stgnn.pt
│       ├── best_nograph.pt
│       ├── pred_stgnn.npy
│       ├── pred_nograph.npy
│       ├── pred_persistence.npy
│       ├── pred_climatology.npy
│       └── y_true_test.npy
│
├── 01_eda.ipynb                # Discharge distributions, seasonality, missingness
├── 02_graph.ipynb              # Adjacency matrix construction, river network map
├── 03_dataset.ipynb            # Feature engineering, sliding windows, splits
├── 04_model.ipynb              # STGNN architecture, forward pass verification
├── 05_train.ipynb              # Training loop, loss curves, baseline predictions
├── 06_evaluate.ipynb           # MAE/RMSE/R², ablation table, forecast plots
│
├── streamlit_app.py            # Interactive dashboard (6 pages)
├── requirements.txt
└── README.md

---

## Running the Streamlit App

After running all notebooks to populate `data/processed/`:

```bash
streamlit run streamlit_app.py
```

The app has six pages:

| Page | Content |
|---|---|
| Overview | Project summary, key findings, architecture table |
| Data Explorer | Interactive discharge time series, summary stats, seasonality |
| River Network | Geographic station map, adjacency heatmap, edge table |
| Forecast Explorer | Per-window 7-day forecast viewer with model overlay |
| Model Evaluation | MAE/RMSE/R² by station and horizon, comparison tables |
| Ablation Study | STGNN vs No-Graph scatter, explanation of null result |

No retraining is needed — the app loads pre-computed artifacts from
`data/processed/` directly.

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

01_eda.ipynb       →  data/processed/daily_discharge.csv
02_graph.ipynb     →  data/processed/adj_norm.npy, station_order.json
03_dataset.ipynb   →  data/processed/X/y tensors, scalers.pkl, config.json
04_model.ipynb     →  architecture definition, forward pass verification
05_train.ipynb     →  best_stgnn.pt, best_nograph.pt, pred_*.npy
06_evaluate.ipynb  →  metrics, plots, ablation table

### 4. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

## Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
networkx
torch>=2.0
streamlit>=1.32
plotly>=5.20

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Temporal encoder | GRU (not LSTM) | Fewer parameters, comparable performance on this task |
| Shared GRU weights | Yes, across all nodes | Forces generalisation, reduces parameters by 10× |
| Adjacency | Topology-based, not correlation | Correlation is shared monsoon signal, not hydrological causality |
| Log transform | log(1+Q) before MinMaxScale | Compresses heavy tail, stabilises gradient flow |
| Scaler fit | Train set only | Strict no-leakage policy into val/test |
| Window drop | Any NaN → drop entire window | Clean training signal; negligible data loss (<1%) |
| Early stopping | Patience=15 on val MSE | Generous enough for LR schedule to take effect first |
| Gradient clipping | max_norm=1.0 | Prevents GRU exploding gradients on flood spikes |

---

## Limitations and Future Work

- **Flood peak underestimation.** MSE loss penalises all errors equally.
  A weighted or quantile loss function would improve extreme event prediction,
  which is precisely the operationally relevant regime.
- **No meteorological forcing.** Rainfall, temperature, and soil moisture
  data would likely improve forecasts significantly, especially beyond day 3
  where antecedent discharge becomes less informative.
- **Daily resolution limits graph utility.** Resampling to hourly or
  sub-daily intervals would introduce non-zero upstream–downstream lags
  and is the single most promising next step for making the GCN component
  useful.
- **Static graph.** The adjacency matrix is fixed from known topology.
  A learned or dynamic adjacency could discover non-topological spatial
  dependencies such as shared groundwater or correlated rainfall patterns.
- **Only 11 years of data.** A longer record would allow evaluation on
  multiple extreme flood years and more robust train/val/test splits.

---

## AUTHOR

Umang Dobhal  
Master's Student | Human Intelligence Systems  
Kyushu Institute of Technology, Japan  
Built out of curiosity — for research.
