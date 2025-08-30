# GeoSteering

A lightweight pipeline for ingesting well `.las` files, normalizing and modeling key drilling signals, forecasting next-step values, computing reservoir contact states and risk, and exporting CSV/JSON/PNG artifacts. You can run it as a CLI or via a FastAPI server.

## 🚀 Features

- **LAS Ingestion** with flexible curve alias matching
- **Normalization** using robust statistics from `models/norm_stats.json`
- **Action Recommendations** via a frozen transformer encoder + DQN
- **Multi-signal Forecasts** (GR/ROP/WOB) via Bi-LSTM (optional)
- **Reservoir Contact States** and risk scoring
- **Multiple Export Formats** (CSV, JSON, and PNG plots) to `outputs/`

## 📁 Project Structure

```
Geo/
├── run_geosteering.py    # CLI entry point and end-to-end processing
├── api.py                # FastAPI app exposing /process endpoint
├── models/               # Required model files and normalization stats
├── outputs/              # Generated reports and plots
├── requirements.txt      # Python dependencies
└── openapi.yaml         # API specification
```

## ⚙️ Requirements

- **Python**: 3.10+ recommended
- **OS**: macOS tested (darwin 24.5.0), Linux/Windows likely fine for API/CLI
- **GPU**: Optional; uses CPU if CUDA is unavailable

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-github-repo-url>
   cd Geo
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 📊 Model Files

Place these required files in `models/` (or point via `MODEL_DIR`):

- `norm_stats.json` (required)
- `pretrained_transformer_encoder.pth` (required)
- `dqn_agent_model_offline.pth` (required)
- `bilstm_baseline.pth` (optional but enables forecasts for GR/ROP/WOB and downstream metrics)

> **Note**: If `bilstm_baseline.pth` is missing, GR/ROP/WOB forecasts and GR-based metrics are skipped and processing returns no results.

## 🔧 Environment Variables

- `MODEL_DIR` (default: `Geo/models`)
- `SAVE_DIR` (default: `Geo/outputs`)

Example:
```bash
export MODEL_DIR=/path/to/models
export SAVE_DIR=/path/to/outputs
```

## 🖥️ Running the CLI

Runs a macOS file dialog to select `.las` files or falls back to terminal input.

```bash
python run_geosteering.py
```

### Outputs
- **CSV per well**: `outputs/f3_contact_<WELL>.csv`
- **Plot PNG per well**: `outputs/f3_dashboard_<WELL>.png`
- **JSON per well**: `outputs/f3_json_<WELL>.json`
- **Index JSON across wells**: `outputs/f3_all_outputs.json`

## 🌐 Running the API

### Start server (from project root)
```bash
uvicorn api:app --reload
```

### Health check
```bash
curl http://localhost:8000/health
```

### Process a `.las` file
```bash
curl -X POST http://localhost:8000/process \
  -F "las_file=@/path/to/well.las" \
  -F "gr_target=60" \
  -F "gr_band=15" \
  -F "show_samples=1000" \
  -F "save_png=true" \
  -F "save_csv=true" \
  -F "save_json=true"
```

### Response contains
- `summary` (risk percentiles, final contact %, counts)
- `json_path` (written artifact)
- `breakdown` (state counts)
- `well` (name)

> **Note**: If processing yields no results (e.g., missing GR or Bi-LSTM), API returns HTTP 422.

## 📈 Accepted LAS Curves and Aliases

Your LAS file should include, at minimum, `MD` and `GR`. Aliases (case-insensitive) are recognized:

| Category | Primary | Aliases |
|----------|---------|---------|
| **Depth** | MD | `MD`, `DEPT`, `DEPTH` (falls back to LAS index if missing) |
| **Gamma Ray** | GR | `GR`, `GAMMA` |
| **Resistivity** | RT | `RT`, `RES`, `ILD`, `ILM`, `AT90`, `AT10`, `RDEP`, `RLA\d+` |
| **Density** | RHOB | `RHOB`, `RHOZ`, `DEN` |
| **Neutron** | NPHI | `NPHI`, `NPOR`, `NEUT` |
| **Rate of Penetration** | ROP | `ROP`, `R OP`, `RATE OF PENET...` |
| **Weight on Bit** | WOB | `WOB`, `W ON BIT`, `WTBIT` |
| **Torque** | TORQ | `TORQ`, `TORQUE` |
| **RPM** | RPM | `RPM`, `REV PM` |
| **Pressure** | PRESS | `PRESS`, `PRES`, `SPP` |
| **Flow** | FLOW | `FLOW`, `Q MUD`, `QIN` |
| **Inclination** | INCL | `INCL`, `INCLINATION`, `DIP` |
| **Azimuth** | AZI | `AZI`, `AZIMUTH` |
| **True Vertical Depth** | TVD | `TVD`, `TVDSS` (defaults to MD if missing) |

## ⚙️ Parameters

### API `/process`
- `las_file` (required): uploaded `.las`
- `gr_target` (float, default 60.0)
- `gr_band` (float, default 15.0)
- `show_samples` (int | omit for `'all'`)
- `save_png`, `save_csv`, `save_json` (bool)

### Function `process_well(...)`
- `las_path` (str, required)
- `GR_TARGET`, `GR_BAND` (floats)
- `show_samples` (`'all'` | int)
- `plan_md_by_window` (np.ndarray | None)
- `save_png`, `save_csv`, `save_json` (bool)

## 📤 Outputs Overview

- **CSV**: window-level features, forecasts, actions, risk, and plan deltas
- **PNG**: GR series with target band and action deltas
- **JSON**: complete timeseries, summaries, and artifact paths
- **Index JSON**: aggregate across multiple processed wells

## 🚨 Troubleshooting

- **Missing GR**: ensure your GR curve uses a recognized mnemonic (see aliases above)
- **`Bi-LSTM: not found`**: add `models/bilstm_baseline.pth`
- **422 from API**: usually due to missing GR or missing Bi-LSTM model
- **Empty outputs**: confirm `MODEL_DIR` contains all required files and `norm_stats.json`

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support information here]
