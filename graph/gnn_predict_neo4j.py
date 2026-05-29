"""
GNN Prediction - FINAL VERSION (offline, dùng cache)
======================================================
Node feature cache đã có sẵn: gnn_node_features_cache.csv
Tính fraud_nb_count từ graph edges + Boosting predictions.

Pipeline:
  1. Load cache (không cần Neo4j nữa)
  2. Build adjacency từ graph/results/all_scores.csv hoặc mining-result
  3. Tính fraud_nb_count dùng boosting predictions làm fraud seed
  4. Build 19-dim feature matrix → GNN inference → xuất gnn_predictions.csv
"""

import os, sys, io, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ─── Paths ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output-trustscore"
MODEL_PATH = OUTPUT_DIR / "gnn_model.pkl"
OUTPUT_CSV = OUTPUT_DIR / "gnn_predictions.csv"
CACHE_CSV  = OUTPUT_DIR / "gnn_node_features_cache.csv"
BOOSTING   = OUTPUT_DIR / "final_output.csv"

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 50000


# ─── Model ──────────────────────────────────────────────
class FraudGNN(nn.Module):
    def __init__(self, in_dim=19, hidden=128, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.Linear(in_dim, hidden)
        self.bn0  = nn.BatchNorm1d(hidden)
        if HAS_PYG:
            self.sage1 = SAGEConv(hidden, hidden)
            self.sage2 = SAGEConv(hidden, hidden)
        self.bn_s = nn.BatchNorm1d(hidden)
        if HAS_PYG:
            self.gat1 = GATConv(hidden, hidden // heads, heads=heads, concat=True)
            self.gat2 = GATConv(hidden, hidden // heads, heads=heads, concat=True)
        self.bn_g = nn.BatchNorm1d(hidden)
        self.mlp  = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x, edge_index=None):
        h = F.relu(self.bn0(self.proj(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if HAS_PYG and edge_index is not None:
            s = F.relu(self.sage1(h, edge_index))
            s = F.dropout(s, p=self.dropout, training=self.training)
            s = F.relu(self.bn_s(self.sage2(s, edge_index)))
            g = F.relu(self.gat1(h, edge_index))
            g = F.dropout(g, p=self.dropout, training=self.training)
            g = F.relu(self.bn_g(self.gat2(g, edge_index)))
        else:
            s = F.relu(self.bn_s(h))
            g = F.relu(self.bn_g(h))
        return self.head(torch.cat([s, g, self.mlp(x)], dim=1))


def load_model():
    print(f"[Model] Loading: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    cfg = data.get('model_config', {})
    in_dim, hidden, heads, dropout = (
        cfg.get('in_dim', 19), cfg.get('hidden', 128),
        cfg.get('heads', 4),   cfg.get('dropout', 0.5)
    )
    print(f"  in_dim={in_dim}, threshold={data['threshold']:.4f}")
    model = FraudGNN(in_dim, hidden, heads, dropout)
    sd = data['model_state_dict']
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()}, strict=False)
    model.to(DEVICE).eval()
    return {
        'model':        model,
        'threshold':    data['threshold'],
        'scaler':       data.get('feature_scaler'),
        'cont_indices': data.get('cont_col_indices', []),
        'le_bank':      data.get('le_bank'),
        'le_entity':    data.get('le_entity'),
        'in_dim':       in_dim,
    }


def compute_fraud_nb_offline(node_df):
    """
    Tính fraud_nb_count từ all_scores.csv (chứa src, dst) + Boosting predictions.
    Boosting predictions dùng làm fraud seed (fraud_prob > 0.5: ~5,348 accounts).
    """
    print("\n[FraudNB] Computing fraud neighbor features offline...")

    # --- Lấy fraud seeds từ Boosting ---
    boost = pd.read_csv(BOOSTING, usecols=['account_id', 'fraud_probability', 'prediction'])
    boost = boost.rename(columns={'account_id': 'id'})

    # Dùng prediction=1 AND fraud_prob > 0.5 để lấy confident fraud seeds
    fraud_seeds = set(boost.loc[boost['fraud_probability'] > 0.5, 'id'].values)
    print(f"  Fraud seeds (prob > 0.5): {len(fraud_seeds):,}")

    # --- Load adjacency từ edges_export.csv (đã export từ Neo4j) ---
    edges_path = OUTPUT_DIR / "edges_export.csv"

    if edges_path.exists():
        print(f"  Loading edges from: {edges_path}  (may take ~30s)")
        try:
            edges = pd.read_csv(edges_path, dtype=str)   # đọc cả 2 cột là str
            edges.columns = ['src', 'dst']
            print(f"  Loaded {len(edges):,} edges")
        except Exception as e:
            print(f"  Error loading edges: {e}")
            edges = None
    else:
        print(f"  [WARN] edges_export.csv not found at {edges_path}")
        edges = None

    if edges is None:
        print("  [WARN] Không tìm thấy file edges. fraud_nb_count = 0 (fallback)")
        node_df['fraud_nb_count'] = 0.0
        node_df['fraud_nb_ratio'] = 0.0
        return node_df

    # --- Đếm fraud senders cho mỗi dst ---
    # Với mỗi edge (src → dst): nếu src là fraud seed → dst có 1 fraud neighbor
    print("  Counting fraud neighbors (src is fraud_seed → dst gets +1)...")
    fraud_seeds_str = {str(x) for x in fraud_seeds}   # đảm bảo str
    fraud_edges = edges[edges['src'].isin(fraud_seeds_str)]
    print(f"  Fraud-origin edges: {len(fraud_edges):,}")

    fraud_nb_count = fraud_edges.groupby('dst').size().reset_index(name='fraud_nb_count')
    fraud_nb_count.columns = ['id', 'fraud_nb_count']
    fraud_nb_count['id'] = fraud_nb_count['id'].astype(str)

    # Merge vào node_df (ép cả 2 về str trước)
    node_df['id'] = node_df['id'].astype(str)
    node_df = node_df.merge(fraud_nb_count, on='id', how='left')
    node_df['fraud_nb_count'] = node_df['fraud_nb_count'].fillna(0).astype(float)

    # Tính ratio
    in_deg = node_df['in_degree'].fillna(0).astype(float) + 1e-6
    node_df['fraud_nb_ratio'] = (node_df['fraud_nb_count'] / in_deg).clip(0, 1)

    nb_nz = (node_df['fraud_nb_count'] > 0).sum()
    print(f"  Accounts with fraud neighbors : {nb_nz:,}")
    print(f"  fraud_nb_count max            : {node_df['fraud_nb_count'].max():.0f}")
    print(f"  fraud_nb_ratio max            : {node_df['fraud_nb_ratio'].max():.4f}")
    return node_df


def build_and_predict(node_df, model_data):
    model        = model_data['model']
    threshold    = model_data['threshold']
    scaler       = model_data['scaler']
    cont_indices = model_data['cont_indices']
    le_bank      = model_data['le_bank']
    le_entity    = model_data['le_entity']
    in_dim       = model_data['in_dim']

    N = len(node_df)
    print(f"\n[Build] {in_dim}-dim feature matrix for {N:,} accounts...")

    X = np.zeros((N, in_dim), dtype=np.float32)

    def col(name, default=0.0):
        if name in node_df.columns:
            return node_df[name].fillna(default).astype(float).values
        return np.full(N, default, dtype=np.float32)

    in_deg  = col('in_degree')
    out_deg = col('out_degree')
    u_snd   = col('unique_senders')
    u_rcv   = col('unique_receivers')
    t_snt   = col('total_sent')
    t_rcv   = col('total_received')

    X[:,0]  = in_deg
    X[:,1]  = out_deg
    X[:,2]  = u_snd
    X[:,3]  = u_rcv
    X[:,4]  = t_snt
    X[:,5]  = t_rcv
    X[:,6]  = t_snt  / (t_rcv + 1e-6)
    X[:,7]  = out_deg / (in_deg + 1e-6)
    X[:,8]  = t_snt  / (out_deg + 1)
    X[:,9]  = t_rcv  / (in_deg  + 1)
    X[:,10] = (u_snd + u_rcv) / (in_deg + out_deg + 1e-6)
    X[:,11] = np.log1p(t_snt)
    X[:,12] = np.log1p(t_rcv)
    X[:,13] = np.log1p(X[:,8])
    X[:,14] = np.log1p(X[:,9])
    X[:,15] = col('fraud_nb_count')
    X[:,16] = col('fraud_nb_ratio')

    # Vectorized label encoding (dùng mapping dict thay vì loop)
    if in_dim > 17 and le_bank is not None:
        bank_map = {cls: i for i, cls in enumerate(le_bank.classes_)}
        bank_vals = node_df['bank_id'].fillna(0).astype(str)
        X[:,17] = bank_vals.map(bank_map).fillna(0).astype(np.float32).values

    if in_dim > 18 and le_entity is not None:
        ent_map = {cls: i for i, cls in enumerate(le_entity.classes_)}
        ent_vals = node_df['entity_id'].fillna('').astype(str)
        X[:,18] = ent_vals.map(ent_map).fillna(0).astype(np.float32).values

    # Scale
    if scaler is not None:
        vi = [i for i in cont_indices if i < in_dim]
        X[:, vi] = scaler.transform(X[:, vi])

    print(f"[Predict] Inference on {DEVICE}...")
    all_probs = np.zeros(N, dtype=np.float32)

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        Xb  = torch.tensor(X[start:end], device=DEVICE)
        with torch.no_grad():
            if HAS_PYG:
                nb = end - start
                ei = torch.arange(nb, device=DEVICE).repeat(2, 1)
                logits = model(Xb, ei)
            else:
                logits = model(Xb, None)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs[start:end] = probs
        print(f"  {end:,}/{N:,}  ({end/N*100:.0f}%)")

    preds   = (all_probs >= threshold).astype(int)
    n_fraud = int(preds.sum())
    print(f"\n  Fraud predicted : {n_fraud:,}  ({n_fraud/N*100:.3f}%)")
    print(f"  Prob: min={all_probs.min():.4f}  "
          f"median={np.median(all_probs):.4f}  "
          f"max={all_probs.max():.4f}")

    return pd.DataFrame({
        'id':             node_df['id'].values,
        'gnn_fraud_prob': all_probs,
        'gnn_prediction': preds,
    })


def main():
    print("=" * 60)
    print("GNN PREDICTION - FINAL (offline cache + boosting seeds)")
    print("=" * 60)

    md = load_model()

    # Step 1: Load node cache (đã fetch từ Neo4j trước đó)
    print(f"\n[Cache] Loading: {CACHE_CSV}")
    node_df = pd.read_csv(CACHE_CSV)
    print(f"  Loaded {len(node_df):,} nodes, cols: {node_df.columns.tolist()}")

    # Step 2: Tính fraud_nb offline
    node_df = compute_fraud_nb_offline(node_df)

    # Step 3: Predict
    result_df = build_and_predict(node_df, md)

    # Save
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Done] Saved: {OUTPUT_CSV}  ({len(result_df):,} rows)")

    print("\n── TOP 10 FRAUD PROBABILITY ──")
    print(result_df.nlargest(10, 'gnn_fraud_prob').to_string(index=False))
    print("\n── SAMPLE LOWEST (normal) ──")
    print(result_df.nsmallest(5, 'gnn_fraud_prob')[['id','gnn_fraud_prob','gnn_prediction']].to_string(index=False))
    print("\n[DONE] Chạy tiếp: python graph/graph_mining.py")


if __name__ == "__main__":
    main()
