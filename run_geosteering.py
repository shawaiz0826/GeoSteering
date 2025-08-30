"""
F3 "All-wells" Dashboard (macOS upload version)

Ingest LAS files, normalize curves with robust stats, run pretrained models to
produce forecasts (GR/ROP/WOB), DQN geosteering actions, contact states, and a
lightweight risk score. Exports per-well CSV/PNG/JSON and an overall index.
"""
# =============================
# F3 "All-wells" Dashboard  
# =============================

import os, re, glob, json, math, warnings, datetime, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings("ignore")

# ---------- Local Paths (macOS-friendly) ----------
# MODEL_DIR: folder with your model files + norm_stats.json
# SAVE_DIR:  outputs (CSV/PNG/JSON + index)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "outputs")

MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
SAVE_DIR  = os.environ.get("SAVE_DIR",  DEFAULT_SAVE_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("MODEL_DIR:", MODEL_DIR)
print("SAVE_DIR :", SAVE_DIR)

# ---------- Prompt user to select .las files ----------
def get_las_paths_from_user() -> List[str]:
    """
    On macOS, opens a native file dialog to pick one or more .las files.
    Falls back to terminal input of a file OR directory.
    """
    paths: List[str] = []
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        selected = filedialog.askopenfilenames(
            title="Select one or more .las files",
            filetypes=[("LAS files", "*.las")]
        )
        paths = list(selected)
        if paths:
            return paths
    except Exception as e:
        print(f"(file dialog not available: {e})")

    # Fallback: terminal prompt
    p = input("Enter the path to a .las file or a directory containing .las files: ").strip()
    if os.path.isdir(p):
        paths = glob.glob(os.path.join(p, "**", "*.las"), recursive=True)
    elif os.path.isfile(p) and p.lower().endswith(".las"):
        paths = [p]
    else:
        print("No valid path provided.")
    return paths

# ---------- Feature contract ----------
FEATURES = ["MD","TVD","GR","ROP","WOB","Torque","RPM","Pressure","Flow",
            "Inclination","Azimuth","Resistivity","Density","Neutron"]
F       = len(FEATURES)
SEQ_LEN = 64
VAL_DIM = F

# === features we use for risk (z-score layer)
RISK_FEATURES = ["Torque","RPM","Pressure","ROP","WOB"]
RISK_IDX = [FEATURES.index(k) for k in RISK_FEATURES]

# ---------- Load normalization stats ----------
with open(os.path.join(MODEL_DIR, "norm_stats.json"), "r") as f:
    stats_meta = json.load(f)
stats = stats_meta["stats"]

# ---------- Model defs (must match training) ----------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings for Transformers."""
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MaskedAutoencoder(nn.Module):
    """Transformer encoder (pretrained as masked autoencoder) producing latents.

    During inference we use only the encoder path. Heads remain for compatibility
    with training checkpoints.
    """
    def __init__(self, in_dim, n_value_feats, d_model=128, depth=4, heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos   = PositionalEncoding(d_model)
        enc_layer  = nn.TransformerEncoderLayer(d_model, heads, d_model*mlp_ratio, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.recon_head = nn.Linear(d_model, n_value_feats)
        self.md_head    = nn.Linear(d_model, 1)
    def forward(self, x, pretrain=False):
        h = self.encoder(self.pos(self.embed(x)))
        return h if not pretrain else (self.recon_head(h), self.md_head(h))

class EncoderWrapper(nn.Module):
    """Freeze encoder and mean-pool sequence outputs to [B, 128] latents."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        for p in self.enc.parameters(): p.requires_grad = False
    def forward(self, x_seq):  # [B,L,3F]
        with torch.no_grad():
            h = self.enc(x_seq, pretrain=False)  # [B,L,128]
            return h.mean(dim=1)                 # [B,128]

class QNet(nn.Module):
    """Small MLP mapping encoder state to Q-values over discrete actions."""
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, s): return self.net(s)

class BiLSTMForecast(nn.Module):
    """Bidirectional LSTM forecaster for next-step GR/ROP/WOB (normalized)."""
    def __init__(self, in_dim, hid=256, out_dim=3, depth=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid//2, num_layers=depth, dropout=dropout,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(),
                                  nn.Linear(hid, out_dim))
    def forward(self, x):  # [B,L,2F]
        h,_ = self.lstm(x)
        return self.head(h[:,-1,:])

# ---------- Load models (from MODEL_DIR) ----------
encoder = MaskedAutoencoder(in_dim=3*F, n_value_feats=F, d_model=128, depth=4, heads=4).to(device)
encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pretrained_transformer_encoder.pth"),
                                   map_location=device))
encoder.eval()
enc_frozen = EncoderWrapper(encoder).to(device)
print("Encoder loaded.")

Q_ACTIONS = np.array([-3,-2,-1,0,+1,+2,+3], dtype=int)
qnet = QNet(128, len(Q_ACTIONS)).to(device)
qnet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dqn_agent_model_offline.pth"),
                                map_location=device))
qnet.eval()
print("DQN loaded.")

bilstm = BiLSTMForecast(in_dim=2*F, hid=256, out_dim=3, depth=2, dropout=0.1).to(device)
try:
    sd = torch.load(os.path.join(MODEL_DIR, "bilstm_baseline.pth"), map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.","",1) if k.startswith("module.") else k: v for k,v in sd.items()}
    bilstm.load_state_dict(sd, strict=False)
    bilstm.eval()
    HAS_BILSTM = True
    print("Bi-LSTM: loaded")
except FileNotFoundError:
    HAS_BILSTM = False
    print("Bi-LSTM: not found (skipping forecasts)")

# ---------- LAS ingestion ----------
import lasio
import numpy.ma as ma

CURVE_ALIASES: Dict[str, str] = {
    "MD": r"^(MD|DEPT|DEPTH)$",
    "GR": r"^(GR|GAMMA)$",
    "Resistivity": r"^(RT|RES|ILD|ILM|AT90|AT10|RDEP|RLA\d+)$",
    "Density": r"^(RHOB|RHOZ|DEN)$",
    "Neutron": r"^(NPHI|NPOR|NEUT)$",
    "ROP": r"^(ROP|R[_\- ]?OP|RATE[_\- ]?OF[_\- ]?PENET.*)$",
    "WOB": r"^(WOB|W[_\- ]?ON[_\- ]?BIT|WTBIT)$",
    "Torque": r"^(TORQ|TORQUE)$",
    "RPM": r"^(RPM|REV[_\- ]?PM)$",
    "Pressure": r"^(PRESS|PRES|SPP)$",
    "Flow": r"^(FLOW|Q[_\- ]?MUD|QIN)$",
    "Inclination": r"^(INCL|INCLINATION|DIP)$",
    "Azimuth": r"^(AZI|AZIMUTH)$",
    "TVD": r"^(TVD|TVDSS)$",
}

def _to_np_array(curve_obj):
    """Convert LAS curve (or array-like) to float64 NumPy array with NaNs for non-finite."""
    data = getattr(curve_obj, "data", curve_obj)
    data = ma.getdata(data)
    arr = np.asarray(data, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr

def pick_curve(las, target: str):
    """Pick curve by alias regex; return NumPy array or None if not found."""
    pat = re.compile(CURVE_ALIASES[target], re.I)
    for c in las.curves:
        mnem = (c.mnemonic or "").strip()
        if pat.match(mnem):
            try: curve_obj = las.get_curve(mnem)
            except Exception: curve_obj = las.curves[mnem]
            return _to_np_array(curve_obj)
    return None

def load_f3_well(las_path: str) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Read a LAS file, standardize curve names, pad lengths, and sort by MD."""
    las = lasio.read(las_path, ignore_header_errors=True)
    found, data = {}, {}
    for t in ["MD","TVD","GR","ROP","WOB","Torque","RPM","Pressure","Flow",
              "Inclination","Azimuth","Resistivity","Density","Neutron"]:
        v = pick_curve(las, t)
        if v is not None:
            data[t] = v; found[t] = "✓"
        else:
            found[t] = "—"
    if "MD" not in data:
        try:
            data["MD"] = _to_np_array(las.index) if hasattr(las, "index") else _to_np_array(las.depths)
            found["MD"] = "✓(index)"
        except Exception:
            raise RuntimeError(f"MD not found in {os.path.basename(las_path)}")

    max_len = max(len(v) for v in data.values())
    for k, v in list(data.items()):
        if len(v) < max_len:
            pad = np.full(max_len - len(v), np.nan, dtype=float)
            data[k] = np.concatenate([v, pad], axis=0)

    df = pd.DataFrame(data)
    if "TVD" not in df.columns:
        df["TVD"] = df["MD"].values
    df = df.sort_values("MD").drop_duplicates(subset=["MD"]).reset_index(drop=True)
    df["__source_file"] = os.path.basename(las_path)
    return df, found

# ---------- Normalization ----------
def normalize_block(df: pd.DataFrame) -> np.ndarray:
    """Normalize to robust z-scores and assemble [N, 3F] (values/masks/tokens)."""
    for c in FEATURES:
        if c not in df.columns: df[c] = np.nan
    vals, masks = [], []
    for c in FEATURES:
        s   = df[c].astype("float32").values
        med = float(stats[c]["median"]); iqr = float(max(stats[c]["iqr"], 1e-6))
        x   = (s - med) / iqr
        vals.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        masks.append((~np.isnan(s)).astype("float32"))
    V = np.clip(np.stack(vals, axis=1), -10.0, 10.0)
    M = np.stack(masks, axis=1).astype("float32")
    Z = np.zeros_like(V, dtype=np.float32)  # mask-token channel (none at inference)
    return np.concatenate([V, M, Z], axis=1)  # [N, 3F]

# ---------- Sliding-window utilities ----------
class SeqIndex:
    """Sliding window indexer for sequences of length L over N samples."""
    def __init__(self, N, L):
        self.starts = list(range(0, max(N-L+1, 0)))
        self.ends   = [s+L for s in self.starts]
    def __len__(self): return len(self.starts)
    def slice(self,i): return self.starts[i], self.ends[i]

@torch.no_grad()
def dqn_actions_for_X3F(X3F: np.ndarray, batch=512) -> Tuple[np.ndarray, np.ndarray]:
    """Infer DQN action indices and return encoder latents per window.

    Returns (actions_idx [W], latents [W, 128]).
    """
    idx = SeqIndex(len(X3F), SEQ_LEN)
    acts, latents = [], []
    for b in range(0, len(idx), batch):
        slab = slice(b, min(b+batch, len(idx)))
        S,E = idx.starts[slab], idx.ends[slab]
        X = np.stack([X3F[s:e] for s,e in zip(S,E)], 0)  # [B,L,3F]
        x = torch.tensor(X, dtype=torch.float32, device=device)
        z = enc_frozen(x)                 # [B,128]
        q = qnet(z)                       # [B,A]
        a = q.argmax(1).detach().cpu().numpy()
        acts.append(a)
        latents.append(z.detach().cpu().numpy())
    if not acts: return np.zeros(0, dtype=int), np.zeros((0,128), np.float32)
    return np.concatenate(acts,0), np.concatenate(latents,0)

@torch.no_grad()
def bilstm_multi_forecast_for_X3F(X3F: np.ndarray, batch=512):
    """Predict next-step GR/ROP/WOB with BiLSTM if available; else return {}."""
    if not HAS_BILSTM: return {}
    X2F = np.concatenate([X3F[:, :F], X3F[:, F:2*F]], axis=1)
    idx = SeqIndex(len(X2F), SEQ_LEN)
    outs = {"GR": [], "ROP": [], "WOB": []}
    gts  = {"GR": [], "ROP": [], "WOB": []}
    i_map = {"GR": FEATURES.index("GR"), "ROP": FEATURES.index("ROP"), "WOB": FEATURES.index("WOB")}
    denorm = {k: (float(stats[k]["median"]), float(max(stats[k]["iqr"], 1e-6))) for k in ["GR","ROP","WOB"]}
    for b in range(0, len(idx), batch):
        slab = slice(b, min(b+batch, len(idx)))
        S,E = idx.starts[slab], idx.ends[slab]
        X = np.stack([X2F[s:e] for s,e in zip(S,E)], 0)  # [B,L,2F]
        x = torch.tensor(X, dtype=torch.float32, device=device)
        y = bilstm(x).detach().cpu().numpy()            # [B,3] normalized
        for j,(k, jF) in enumerate(i_map.items()):
            med, iqr = denorm[k]
            pred_norm = y[:, j]
            gt_norm   = X[:, -1, :F][:, jF]
            outs[k].append(pred_norm * iqr + med)
            gts[k].append (gt_norm   * iqr + med)
    for k in outs:
        outs[k] = np.concatenate(outs[k],0) if len(outs[k]) else np.array([])
        gts[k]  = np.concatenate(gts[k],0)  if len(gts[k])  else np.array([])
    return {"pred": outs, "true": gts}

# Persistence baseline for Torque/Pressure
def persistence_forecast_for_X3F(X3F: np.ndarray, feature_name: str):
    """Simple baseline: next value equals last observed value per window."""
    j = FEATURES.index(feature_name)
    idx = SeqIndex(len(X3F), SEQ_LEN)
    med = float(stats[feature_name]["median"]); iqr = float(max(stats[feature_name]["iqr"], 1e-6))
    y_true, y_pred = [], []
    for i in range(len(idx)):
        s,e = idx.slice(i)
        last_norm = X3F[e-1, j]
        phys = last_norm * iqr + med
        y_true.append(phys)
        y_pred.append(phys)      # persistence
    return np.array(y_pred), np.array(y_true)

# ---------- Contact-state logic ----------
def contact_states(gr_phys: np.ndarray, gr_pred_next: np.ndarray | None,
                   target: float = 60.0, band: float = 15.0,
                   smooth_win: int = 11):
    """Assign contact-state labels vs GR target band using smoothed signal and trend."""
    gr_s = pd.Series(gr_phys).rolling(smooth_win, center=True, min_periods=1).median().values
    d = gr_s - target
    status = np.full(len(gr_s), "outside", dtype=object)
    inside = np.abs(d) <= band
    status[inside] = "in_zone"
    grad = np.gradient(d)
    trend = grad.copy()
    if gr_pred_next is not None:
        f_ok = np.isfinite(gr_pred_next)
        delta_f = np.zeros_like(gr_s)
        delta_f[f_ok] = gr_pred_next[f_ok] - gr_s[f_ok]
        trend = 0.7 * grad + 0.3 * delta_f
    approaching = inside & (d * trend > 0)
    status[approaching] = "approaching_exit"
    exiting = (~inside) & (d * trend > 0)
    status[exiting] = "exiting"
    breakdown = {k: int(np.sum(status == k)) for k in ["outside","in_zone","approaching_exit","exiting"]}
    return status, breakdown

# Lightweight risk score (0-100)
def compute_risk_scores(X3F: np.ndarray,
                        ytrue_gr: np.ndarray,
                        yhat_gr: np.ndarray,
                        gr_target: float = 60.0,
                        gr_band: float   = 15.0):
    """Compute window-level risk (0-100), severity labels, and top driver signals."""
    idx = SeqIndex(len(X3F), SEQ_LEN)
    iqr_gr = float(max(stats["GR"]["iqr"], 1e-6))
    risks, sevs, top_feats = [], [], []
    for i in range(len(idx)):
        s,e = idx.slice(i)
        Vend = X3F[e-1, :F]
        z_abs = np.abs(Vend[RISK_IDX])
        z_agg = float(np.mean(np.clip(z_abs, 0, 6)))
        order = np.argsort(-z_abs)[:3]
        top = [f"{RISK_FEATURES[k]}:{z_abs[k]:.1f}σ" for k in order]
        top_feats.append(top)
        fe = float(abs(yhat_gr[i] - ytrue_gr[i]) / iqr_gr); fe = min(fe, 6.0)
        d = abs(ytrue_gr[i] - gr_target) - gr_band
        prox = max(d, 0.0) / max(gr_band, 1e-6); prox = min(prox, 3.0)
        linear = 0.6*z_agg + 0.5*fe + 0.8*prox
        risk = 100.0 * math.tanh(linear / 3.0)
        risks.append(risk)
        if   risk >= 85: sevs.append("critical")
        elif risk >= 70: sevs.append("warning")
        elif risk >= 50: sevs.append("info")
        else:            sevs.append("normal")
    return np.array(risks), np.array(sevs, dtype=object), top_feats

# ---------- Plotting ----------
Q_ACTIONS = np.array([-3,-2,-1,0,+1,+2,+3], dtype=int)

def plot_dashboard(
    well_name, md, gr_phys, gr_pred_next=None, actions_idx=None,
    target=60.0, band=15.0, show_samples='all', save_png_path=None,
    plan_md=None, delta_md=None
):
    """Plot GR with target band and Δ-inclination actions; optionally save PNG."""
    Nw = len(gr_phys)
    K = Nw if (show_samples in ['all', None]) else max(1, min(int(show_samples), Nw))
    xs = np.arange(Nw)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(xs[:K], gr_phys[:K], label="GR (gAPI)")
    if gr_pred_next is not None and np.isfinite(gr_pred_next[:K]).any():
        ax1.plot(xs[:K], gr_pred_next[:K], "--", label="GR next-step pred")
    ax1.axhline(target, ls="--", label="GR target")
    ax1.axhspan(target - band, target + band, alpha=0.2, label="target band")
    ax1.set_ylabel("gAPI")
    ttl = f"{well_name} · first {K} samples"
    if delta_md is not None and len(delta_md) >= K:
        last_delta = float(delta_md[min(K-1, len(delta_md)-1)])
        ttl += f" · ΔMD vs plan (last) = {last_delta:+.1f} m"
    ax1.set_title(ttl); ax1.legend(loc="upper right")
    if actions_idx is not None and len(actions_idx):
        deltas = Q_ACTIONS[np.asarray(actions_idx[:K]).astype(int)]
    else:
        deltas = np.zeros(K, dtype=float)
    ax2.plot(xs[:K], deltas, linewidth=0.8)
    ax2.plot(xs[:K], deltas, linestyle="None", marker="o", markersize=2, alpha=0.6)
    ax2.axhline(0.0, color="k", linewidth=0.5)
    ax2.set_ylabel("Δ inc (deg)"); ax2.set_xlabel("sample idx")
    plt.tight_layout()
    if save_png_path: plt.savefig(save_png_path, dpi=150)
    plt.show()

# ---------- Helpers for JSON ----------
def _to_py_list(x):
    """Convert arrays/tuples to Python lists for JSON serialization."""
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (list, tuple)): return list(x)
    return x

def _write_json(path, payload):
    """Write a JSON file at `path` with indentation for readability."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# ---------- End-to-end per well (writes JSON) ----------
def process_well(las_path: str,
                 GR_TARGET=60.0, GR_BAND=15.0,
                 show_samples=1000, save_png=True, save_csv=True,
                 plan_md_by_window: Optional[np.ndarray] = None,
                 print_predictions: bool = True,
                 print_limit: Optional[int] = None,
                 save_json: bool = True):
    """Process a single LAS path end-to-end and emit artifacts and summary dict."""
    df, found = load_f3_well(las_path)
    print(f"\nLoaded {os.path.basename(las_path)} | mapped: {found}")
    if "GR" not in df.columns:
        print("No GR curve → skipping.")
        return None

    X3F = normalize_block(df)
    a_idx, lat = dqn_actions_for_X3F(X3F)
    multi = bilstm_multi_forecast_for_X3F(X3F)
    yhat_tq, ytru_tq = persistence_forecast_for_X3F(X3F, "Torque")
    yhat_pp, ytru_pp = persistence_forecast_for_X3F(X3F, "Pressure")

    if not multi:
        print("No Bi-LSTM — using persistence only for Torque/Pressure. Skipping GR-based metrics.")
        return None

    yhat_gr = multi["pred"]["GR"];  ytru_gr = multi["true"]["GR"]
    yhat_rop= multi["pred"]["ROP"]; ytru_rop= multi["true"]["ROP"]
    yhat_wob= multi["pred"]["WOB"]; ytru_wob= multi["true"]["WOB"]

    Nw = yhat_gr.shape[0]
    md_end = df["MD"].values[SEQ_LEN-1 : SEQ_LEN-1 + Nw]
    well = os.path.basename(las_path)
    src  = df["__source_file"].iloc[0] if "__source_file" in df.columns else os.path.basename(las_path)

    states, breakdown = contact_states(ytru_gr, yhat_gr, target=GR_TARGET, band=GR_BAND)
    in_contact = (states == "in_zone")
    contact_pct_cum = 100.0 * np.cumsum(in_contact.astype(float)) / (np.arange(Nw) + 1)
    risks, sevs, top_feats = compute_risk_scores(X3F, ytru_gr, yhat_gr, gr_target=GR_TARGET, gr_band=GR_BAND)

    if plan_md_by_window is not None and len(plan_md_by_window) >= Nw:
        delta_md = md_end - plan_md_by_window[:Nw]; plan_used = True
    else:
        delta_md = np.full(Nw, np.nan);               plan_used = False

    # CSV (optional)
    csv_path = ""
    if save_csv:
        out_rows = pd.DataFrame({
            "well": well,
            "window_idx": np.arange(Nw),
            "MD_end": md_end,
            "GR_true": ytru_gr, "GR_pred_next": yhat_gr,
            "ROP_true": ytru_rop, "ROP_pred_next": yhat_rop,
            "WOB_true": ytru_wob, "WOB_pred_next": yhat_wob,
            "Torque_true": ytru_tq[:Nw], "Torque_pred_next": yhat_tq[:Nw],
            "Pressure_true": ytru_pp[:Nw], "Pressure_pred_next": yhat_pp[:Nw],
            "action_idx": a_idx[:Nw] if a_idx.size else np.full(Nw, -1),
            "delta_deg": Q_ACTIONS[a_idx[:Nw]] if a_idx.size else np.full(Nw, np.nan),
            "state": states,
            "contact_pct_cum": contact_pct_cum,
            "risk": risks,
            "severity": sevs,
            "top_signals": [", ".join(t) for t in top_feats],
            "plan_MD_end": plan_md_by_window[:Nw] if plan_used else np.full(Nw, np.nan),
            "delta_MD_vs_plan": delta_md
        })
        csv_path = os.path.join(SAVE_DIR, f"f3_contact_{well.replace('.las','')}.csv")
        out_rows.to_csv(csv_path, index=False)
        print(f"📄 wrote per-window CSV → {csv_path}")

    # Plot (optional)
    plot_png = ""
    if save_png:
        fig_path = os.path.join(SAVE_DIR, f"f3_dashboard_{well.replace('.las','')}.png")
        plot_dashboard(
            well, df["MD"].values, ytru_gr, yhat_gr, a_idx,
            target=GR_TARGET, band=GR_BAND, show_samples=show_samples,
            save_png_path=fig_path,
            plan_md=plan_md_by_window[:Nw] if plan_used else None,
            delta_md=delta_md if plan_used else None
        )
        plot_png = fig_path

    # Build JSON payload
    summary = {
        "risk_p50": float(np.nanpercentile(risks, 50)),
        "risk_p90": float(np.nanpercentile(risks, 90)),
        "contact_final_pct": float(contact_pct_cum[-1]),
        "breakdown": {k:int(v) for k,v in breakdown.items()},
        "n_windows": int(Nw),
        "plan_overlay": bool(plan_used)
    }
    payload = {
        "meta": {
            "well_name": well,
            "source_file": src,
            "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z"
        },
        "config": {
            "seq_len": SEQ_LEN,
            "features": FEATURES,
            "gr_target": float(GR_TARGET),
            "gr_band": float(GR_BAND)
        },
        "summary": summary,
        "timeseries": {
            "window_idx": _to_py_list(np.arange(Nw)),
            "MD_end": _to_py_list(md_end),
            "forecasts": {
                "GR": {"true": _to_py_list(ytru_gr), "pred_next": _to_py_list(yhat_gr)},
                "ROP":{"true": _to_py_list(ytru_rop), "pred_next": _to_py_list(yhat_rop)},
                "WOB":{"true": _to_py_list(ytru_wob), "pred_next": _to_py_list(yhat_wob)},
                "Torque":{"true": _to_py_list(ytru_tq[:Nw]), "pred_next": _to_py_list(yhat_tq[:Nw])},
                "Pressure":{"true": _to_py_list(ytru_pp[:Nw]), "pred_next": _to_py_list(yhat_pp[:Nw])}
            },
            "actions": {
                "action_idx": _to_py_list(a_idx[:Nw] if a_idx.size else np.full(Nw, -1)),
                "delta_deg": _to_py_list(Q_ACTIONS[a_idx[:Nw]] if a_idx.size else np.full(Nw, np.nan))
            },
            "reservoir": {
                "state": _to_py_list(states),
                "contact_pct_cum": _to_py_list(contact_pct_cum)
            },
            "risk": {
                "risk": _to_py_list(risks),
                "severity": _to_py_list(sevs),
                "top_signals": _to_py_list([", ".join(t) for t in top_feats])
            },
            "performance": {
                "plan_MD_end": _to_py_list(plan_md_by_window[:Nw] if plan_used else np.full(Nw, np.nan)),
                "delta_MD_vs_plan": _to_py_list(delta_md)
            }
        },
        "artifacts": {
            "csv_path": csv_path,
            "plot_png": plot_png
        }
    }

    json_path = ""
    if save_json:
        json_path = os.path.join(SAVE_DIR, f"f3_json_{well.replace('.las','')}.json")
        _write_json(json_path, payload)
        print(f"🧾 wrote per-well JSON → {json_path}")

    print(f"Risk (p50/p90): {summary['risk_p50']:.1f} / {summary['risk_p90']:.1f} · "
          f"Contact % (final): {summary['contact_final_pct']:.1f}% · "
          f"Plan overlay: {'yes' if summary['plan_overlay'] else 'no'}")

    return {"breakdown": breakdown, "json_path": json_path, "summary": summary, "well": well}

# ---------- Optional contact plot ----------
def plot_contact_windows(rep_df, well_name, GR_TARGET=60.0, GR_BAND=15.0, show_samples='all'):
    """Visualize contact-state windows with GR overlays and suggested actions."""
    if show_samples == 'all' or show_samples is None:
        show_n = len(rep_df)
    else:
        show_n = min(int(show_samples), len(rep_df))
    view = rep_df.iloc[:show_n].copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(view["idx"].values, view["GR_true"].values, label="GR (gAPI)")
    if "GR_pred_next" in view.columns and np.isfinite(view["GR_pred_next"]).any():
        ax1.plot(view["idx"].values, view["GR_pred_next"].values, "--", label="GR next-step pred")
    ax1.axhline(GR_TARGET, ls="--", label="GR target")
    ax1.axhspan(GR_TARGET-GR_BAND, GR_TARGET+GR_BAND, alpha=0.2, label="target band")
    for i, st in enumerate(view["status"].values):
        if st == "in_zone":            ax1.axvspan(i-0.5, i+0.5, color=(0.2,0.6,0.9,0.10))
        elif st == "approaching_exit": ax1.axvspan(i-0.5, i+0.5, color=(1.0,0.5,0.0,0.10))
        elif st == "exiting":          ax1.axvspan(i-0.5, i+0.5, color=(1.0,0.0,0.0,0.12))
    ax1.set_ylabel("gAPI"); ax1.set_title(f"Reservoir-contact states (first {show_n} samples): {well_name}")
    ax1.legend(loc="upper right")
    deltas = view["delta_deg"].values if "delta_deg" in view.columns else np.zeros(len(view))
    xs = view["idx"].values
    ax2.plot(xs, deltas, linewidth=0.8)
    ax2.plot(xs, deltas, linestyle="None", marker="o", markersize=2, alpha=0.5)
    ax2.axhline(0.0, color="k", linewidth=0.5)
    ax2.set_ylabel("Δ inc (deg)"); ax2.set_xlabel("sample idx")
    plt.tight_layout(); plt.show()

# ---------- Run on user-provided LAS files & build index (CLI entry) ----------

def main():
    """CLI entrypoint: prompt/select LAS files, process, and build index JSON."""
    las_files = get_las_paths_from_user()
    if not las_files:
        raise RuntimeError("No .las files provided. Please select or specify at least one .las file.")

    overall = {"outside":0, "in_zone":0, "approaching_exit":0, "exiting":0}
    index_items = []

    for lp in las_files:
        res = process_well(lp, GR_TARGET=60.0, GR_BAND=15.0,
                           show_samples='all', save_png=True, save_csv=True,
                           plan_md_by_window=None, save_json=True)
        if not res:
            continue
        bd = res["breakdown"]
        for k in overall: overall[k] += int(bd.get(k, 0))
        index_items.append({
            "well": res["well"],
            "json_path": res["json_path"],
            "summary": res["summary"]
        })

    index_payload = {
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "n_wells": len(index_items),
        "aggregate_state_counts": overall,
        "wells": index_items
    }
    index_path = os.path.join(SAVE_DIR, "f3_all_outputs.json")
    with open(index_path, "w") as f:
        json.dump(index_payload, f, indent=2)

    print("\n=== Aggregate across uploaded wells (window-level) ===")
    print(overall)
    print(f"🗂️  wrote index JSON → {index_path}")


if __name__ == "__main__":
    main()
