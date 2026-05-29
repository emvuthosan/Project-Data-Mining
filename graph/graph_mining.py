import os
import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output-trustscore"
RESULTS_DIR = BASE_DIR / "graph" / "results"

BOOSTING_CSV = OUTPUT_DIR / "final_output.csv"
TRUST_CSV = OUTPUT_DIR / "trust_scores_output.csv"
GNN_CSV = OUTPUT_DIR / "gnn_predictions.csv"  # Output từ GNN Model của Trung

# Config — Ensemble Scoring Weights
# Khi có GNN: 3 nguồn kết hợp
WEIGHTS_WITH_GNN = {
    "boosting": 0.35,
    "gnn": 0.30,
    "trust": 0.35,
}
# Khi không có GNN: 2 nguồn (như cũ)
WEIGHTS_NO_GNN = {
    "boosting": 0.55,
    "trust": 0.45,
}

SUSPICIOUS_THRESHOLD = 0.55
MIN_CRITERIA = 2


# PHASE 1: XÁC ĐỊNH SUSPICIOUS NODES (CSV only)

def load_data():
    """Load và merge dữ liệu từ Boosting + GNN (nếu có) + Trust Score."""
    print("\n" + "=" * 60)
    print("PHASE 1: LOAD & MERGE DATA")
    print("=" * 60)

    # ── Load Boosting output ──────────────────────────────────
    if not BOOSTING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {BOOSTING_CSV}")
    df_boost = pd.read_csv(BOOSTING_CSV)
    df_boost = df_boost.rename(columns={"account_id": "id"})
    print(f"[Boosting]    {len(df_boost):,} accounts loaded")

    # ── Load Trust Score output ───────────────────────────────
    if not TRUST_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {TRUST_CSV}")
    df_trust = pd.read_csv(TRUST_CSV)
    print(f"[Trust Score] {len(df_trust):,} accounts loaded")

    # ── Load GNN output (optional) ────────────────────────────
    has_gnn = False
    df_gnn = None
    if GNN_CSV.exists():
        df_gnn = pd.read_csv(GNN_CSV)
        # Chuẩn hóa tên cột — GNN có thể dùng "account_id" hoặc "id"
        if "account_id" in df_gnn.columns and "id" not in df_gnn.columns:
            df_gnn = df_gnn.rename(columns={"account_id": "id"})
        # Chuẩn hóa tên cột fraud_prob → gnn_fraud_prob
        if "fraud_prob" in df_gnn.columns:
            df_gnn = df_gnn.rename(columns={"fraud_prob": "gnn_fraud_prob"})
        elif "fraud_probability" in df_gnn.columns:
            df_gnn = df_gnn.rename(columns={"fraud_probability": "gnn_fraud_prob"})
        # Chuẩn hóa tên cột prediction → gnn_prediction
        if "prediction" in df_gnn.columns:
            df_gnn = df_gnn.rename(columns={"prediction": "gnn_prediction"})
        has_gnn = True
        print(f"[GNN Model]   {len(df_gnn):,} accounts loaded ✓")
    else:
        print(f"[GNN Model]   Không tìm thấy file → chạy với Boosting + Trust Score")

    # ── Merge tất cả nguồn dữ liệu ──────────────────────────
    # Bắt đầu từ Trust Score (có nhiều cột nhất)
    df = df_trust.merge(
        df_boost[["id", "fraud_probability", "prediction"]],
        on="id", how="outer"
    )

    if has_gnn and df_gnn is not None:
        gnn_cols = ["id", "gnn_fraud_prob"]
        if "gnn_prediction" in df_gnn.columns:
            gnn_cols.append("gnn_prediction")
        df = df.merge(df_gnn[gnn_cols], on="id", how="outer")

    # ── Fill missing values ───────────────────────────────────
    df["fraud_probability"] = df["fraud_probability"].fillna(0.0)
    df["trust_score"] = df["trust_score"].fillna(0.5)
    df["prediction"] = df["prediction"].fillna(0).astype(int)
    if has_gnn:
        df["gnn_fraud_prob"] = df["gnn_fraud_prob"].fillna(0.0)
        if "gnn_prediction" in df.columns:
            df["gnn_prediction"] = df["gnn_prediction"].fillna(0).astype(int)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n[Merged]      {len(df):,} accounts total")
    print(f"  Có fraud_probability (Boosting): {(df['fraud_probability'] > 0).sum():,}")
    print(f"  Có trust_score:                  {(df['trust_score'] > 0).sum():,}")
    if has_gnn:
        print(f"  Có gnn_fraud_prob (GNN):         {(df['gnn_fraud_prob'] > 0).sum():,}")

    return df, has_gnn


def compute_suspicious_score(df, has_gnn=False):
    """
    Tính suspicious_score với xử lý riêng cho accounts có/không có trust data.

    - Có GNN + trust data: dùng 3-source ensemble (Boosting 0.35 + GNN 0.30 + Trust 0.35)
    - Không có GNN, có trust data: dùng 2-source ensemble (Boosting 0.55 + Trust 0.45)
    - Không có trust data: chỉ dùng fraud_probability (và GNN nếu có)
    """
    print("\n" + "-" * 60)
    print("COMPUTING SUSPICIOUS SCORE")
    print("-" * 60)

    has_trust = df["trust_score"] > 0
    n_with = has_trust.sum()
    n_without = (~has_trust).sum()
    print(f"  Accounts có trust data:    {n_with:,}")
    print(f"  Accounts không trust data: {n_without:,}")
    print(f"  GNN data available:        {'Có ✓' if has_gnn else 'Không ✗'}")

    if has_gnn:
        w = WEIGHTS_WITH_GNN
        print(f"\n  Mode: 3-Source Ensemble (Boosting={w['boosting']}, GNN={w['gnn']}, Trust={w['trust']})")

        # Accounts CÓ trust data → 3-source blended score
        has_gnn_data = df["gnn_fraud_prob"] > 0

        # Case 1: Có cả 3 nguồn
        mask_3 = has_trust & has_gnn_data
        df.loc[mask_3, "suspicious_score"] = (
            w["boosting"] * df.loc[mask_3, "fraud_probability"]
            + w["gnn"] * df.loc[mask_3, "gnn_fraud_prob"]
            + w["trust"] * (1 - df.loc[mask_3, "trust_score"])
        )

        # Case 2: Có Boosting + Trust, KHÔNG có GNN
        mask_bt = has_trust & ~has_gnn_data
        w_bt = WEIGHTS_NO_GNN
        df.loc[mask_bt, "suspicious_score"] = (
            w_bt["boosting"] * df.loc[mask_bt, "fraud_probability"]
            + w_bt["trust"] * (1 - df.loc[mask_bt, "trust_score"])
        )

        # Case 3: Có Boosting + GNN, KHÔNG có Trust
        mask_bg = ~has_trust & has_gnn_data
        df.loc[mask_bg, "suspicious_score"] = (
            0.55 * df.loc[mask_bg, "fraud_probability"]
            + 0.45 * df.loc[mask_bg, "gnn_fraud_prob"]
        )

        # Case 4: Chỉ có Boosting (không Trust, không GNN)
        mask_b = ~has_trust & ~has_gnn_data
        df.loc[mask_b, "suspicious_score"] = df.loc[mask_b, "fraud_probability"]

        print(f"    3-source (Boost+GNN+Trust): {mask_3.sum():,}")
        print(f"    2-source (Boost+Trust):     {mask_bt.sum():,}")
        print(f"    2-source (Boost+GNN):       {mask_bg.sum():,}")
        print(f"    1-source (Boost only):      {mask_b.sum():,}")

    else:
        w = WEIGHTS_NO_GNN
        print(f"\n  Mode: 2-Source Ensemble (Boosting={w['boosting']}, Trust={w['trust']})")

        # Accounts CÓ trust data → blended score
        df.loc[has_trust, "suspicious_score"] = (
            w["boosting"] * df.loc[has_trust, "fraud_probability"]
            + w["trust"] * (1 - df.loc[has_trust, "trust_score"])
        )

        # Accounts KHÔNG CÓ trust data → chỉ dùng fraud_probability
        df.loc[~has_trust, "suspicious_score"] = df.loc[~has_trust, "fraud_probability"]

    print(f"\n  Distribution:")
    print(f"    Min={df['suspicious_score'].min():.4f}  "
          f"Median={df['suspicious_score'].median():.4f}  "
          f"Max={df['suspicious_score'].max():.4f}")

    return df


def identify_suspicious_nodes(df, has_gnn=False):
    """
    Lọc node nghi ngờ dựa trên nhiều tiêu chí.

    Khi có GNN → thêm tiêu chí c5 (GNN prediction == 1), tổng 5 tiêu chí.
    - Có trust data: thỏa >= 2 tiêu chí
    - Không có trust data: fraud_probability >= 0.5 (Boosting tự tin)
    """
    print("\n" + "-" * 60)
    print("IDENTIFYING SUSPICIOUS NODES")
    print("-" * 60)

    has_trust = df["trust_score"] > 0

    # ── Tiêu chí chung ────────────────────────────────────────
    c1 = df["suspicious_score"] >= SUSPICIOUS_THRESHOLD
    c2 = df["prediction"] == 1  # Boosting prediction
    c3 = df["trust_level"] == "VERY_LOW" if "trust_level" in df.columns else pd.Series(False, index=df.index)
    c4 = df["anti_trustrank_norm"] >= 0.7 if "anti_trustrank_norm" in df.columns else pd.Series(False, index=df.index)

    # ── Tiêu chí GNN (nếu có) ─────────────────────────────────
    if has_gnn and "gnn_prediction" in df.columns:
        c5 = df["gnn_prediction"] == 1
        df["criteria_met"] = (c1.astype(int) + c2.astype(int) + c3.astype(int)
                              + c4.astype(int) + c5.astype(int))
        total_criteria = 5
        print(f"  Criteria: suspicious_score, Boosting pred, trust_level, anti_trustrank, GNN pred ({total_criteria} total)")
    elif has_gnn and "gnn_fraud_prob" in df.columns:
        # Nếu GNN không có cột prediction, dùng threshold 0.5
        c5 = df["gnn_fraud_prob"] >= 0.5
        df["criteria_met"] = (c1.astype(int) + c2.astype(int) + c3.astype(int)
                              + c4.astype(int) + c5.astype(int))
        total_criteria = 5
        print(f"  Criteria: suspicious_score, Boosting pred, trust_level, anti_trustrank, GNN prob≥0.5 ({total_criteria} total)")
    else:
        df["criteria_met"] = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
        total_criteria = 4
        print(f"  Criteria: suspicious_score, Boosting pred, trust_level, anti_trustrank ({total_criteria} total)")

    # ── Phân loại suspicious ──────────────────────────────────
    # Có trust data: cần >= MIN_CRITERIA
    # Không trust data: cần fraud_probability >= 0.5 VÀ prediction == 1
    #   (nếu có GNN, thêm điều kiện gnn cũng đồng ý)
    df["is_suspicious"] = False
    df.loc[has_trust, "is_suspicious"] = df.loc[has_trust, "criteria_met"] >= MIN_CRITERIA

    if has_gnn and "gnn_fraud_prob" in df.columns:
        # Không có trust data: Boosting + GNN cùng đồng ý
        df.loc[~has_trust, "is_suspicious"] = (
            (df.loc[~has_trust, "fraud_probability"] >= 0.5)
            & (df.loc[~has_trust, "prediction"] == 1)
        ) | (
            # Hoặc GNN tự tin cao (>= 0.7) cũng suspicious
            (df.loc[~has_trust, "gnn_fraud_prob"] >= 0.7)
        )
    else:
        df.loc[~has_trust, "is_suspicious"] = (
            (df.loc[~has_trust, "fraud_probability"] >= 0.5)
            & (df.loc[~has_trust, "prediction"] == 1)
        )

    # ── Risk Level ────────────────────────────────────────────
    def risk_level(row):
        met = row["criteria_met"]
        if has_gnn:
            # 5 tiêu chí → thang mới
            if met >= 5: return "CRITICAL"
            elif met >= 4: return "CRITICAL"
            elif met >= 3: return "HIGH"
            elif met >= 2: return "MEDIUM"
            else: return "LOW"
        else:
            if met >= 4: return "CRITICAL"
            elif met >= 3: return "HIGH"
            elif met >= 2: return "MEDIUM"
            else: return "LOW"

    df["risk_level"] = df.apply(risk_level, axis=1)

    # ── Report ────────────────────────────────────────────────
    suspicious = df[df["is_suspicious"]]
    total = len(df)
    n_sus = len(suspicious)

    print(f"\n  Tổng accounts:      {total:,}")
    print(f"  Suspicious nodes:   {n_sus:,} ({n_sus/total*100:.2f}%)")
    n_sus_trust = len(suspicious[suspicious["trust_score"] > 0])
    n_sus_no_trust = n_sus - n_sus_trust
    print(f"    - Có trust data:    {n_sus_trust:,}")
    print(f"    - Chỉ có Boosting:  {n_sus_no_trust:,}")
    print(f"\n  Phân bố risk_level:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = (df["risk_level"] == level).sum()
        print(f"    {level:<10}: {n:>8,} ({n/total*100:.1f}%)")

    if "label" in df.columns:
        fraud_nodes = df[df["label"] == 1]
        caught = suspicious[suspicious["label"] == 1]
        if len(fraud_nodes) > 0:
            recall = len(caught) / len(fraud_nodes) * 100
            precision = len(caught) / max(n_sus, 1) * 100
            f1 = 2 * precision * recall / max(precision + recall, 1)
            print(f"\n  Validation vs ground truth:")
            print(f"    Fraud nodes:    {len(fraud_nodes):,}")
            print(f"    Detected:       {len(caught):,} (Recall={recall:.1f}%)")
            print(f"    Precision:      {precision:.1f}%")
            print(f"    F1:             {f1:.1f}%")

    return df


# PHASE 2: AML MOTIF DETECTION (Neo4j)

def _get_neo4j_conn():
    """Tạo kết nối Neo4j dùng config từ .env"""
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    sys.path.insert(0, str(BASE_DIR / "graph"))
    from connect import Neo4jConnection
    return Neo4jConnection(uri, user, password)


def detect_fanout(conn, suspicious_ids, min_targets=5):
    """Fan-out: 1 node gửi tiền cho nhiều node (structuring)."""
    print("\n  [Fan-out] Detecting...")
    query = """
        UNWIND $ids AS sid
        MATCH (a:Account {id: sid})-[t:TRANSFER]->(b:Account)
        WITH a, count(DISTINCT b) AS num_targets,
             collect(DISTINCT b.id) AS target_ids,
             sum(t.amount) AS total_amount, count(t) AS tx_count
        WHERE num_targets >= $minTargets
        RETURN a.id AS source_id, num_targets, target_ids,
               total_amount, tx_count
        ORDER BY num_targets DESC
    """
    results = conn.run_query(query, {"ids": suspicious_ids, "minTargets": min_targets})
    print(f"    Found {len(results)} fan-out patterns")
    return results


def detect_fanin(conn, suspicious_ids, min_sources=5):
    """Fan-in: nhiều node gửi tiền về 1 node (aggregation)."""
    print("\n  [Fan-in] Detecting...")
    query = """
        UNWIND $ids AS sid
        MATCH (b:Account)-[t:TRANSFER]->(a:Account {id: sid})
        WITH a, count(DISTINCT b) AS num_sources,
             collect(DISTINCT b.id) AS source_ids,
             sum(t.amount) AS total_amount, count(t) AS tx_count
        WHERE num_sources >= $minSources
        RETURN a.id AS target_id, num_sources, source_ids,
               total_amount, tx_count
        ORDER BY num_sources DESC
    """
    results = conn.run_query(query, {"ids": suspicious_ids, "minSources": min_sources})
    print(f"    Found {len(results)} fan-in patterns")
    return results


def detect_cycles(conn, suspicious_ids, max_length=3):
    """Cycle: A → B → C → ... → A (layering)."""
    print(f"\n  [Cycle] Detecting (max length={max_length})...")
    query = """
        UNWIND $ids AS sid
        MATCH path = (a:Account {id: sid})-[:TRANSFER*2..%d]->(a)
        WITH a, path, length(path) AS cycle_len,
             [r IN relationships(path) | r.amount] AS amounts,
             [n IN nodes(path) | n.id] AS node_chain
        RETURN a.id AS start_node, cycle_len, node_chain, amounts,
               reduce(s=0.0, x IN amounts | s + x) AS total_flow
        ORDER BY cycle_len LIMIT 500
    """ % max_length
    results = conn.run_query(query, {"ids": suspicious_ids})
    print(f"    Found {len(results)} cycle patterns")
    return results


def detect_chains(conn, suspicious_ids, min_length=3, max_length=4):
    """Chain: A → B → C → D (pass-through)."""
    print(f"\n  [Chain] Detecting (length {min_length}-{max_length})...")
    query = """
        UNWIND $ids AS sid
        MATCH path = (a:Account {id: sid})-[:TRANSFER*%d..%d]->(end_node:Account)
        WHERE a <> end_node
        WITH a, end_node, path, length(path) AS chain_len,
             [n IN nodes(path) | n.id] AS node_chain,
             [r IN relationships(path) | r.amount] AS amounts
        RETURN a.id AS start_node, end_node.id AS end_node,
               chain_len, node_chain, amounts
        ORDER BY chain_len DESC LIMIT 500
    """ % (min_length, max_length)
    results = conn.run_query(query, {"ids": suspicious_ids})
    print(f"    Found {len(results)} chain patterns")
    return results


def detect_scatter_gather(conn, suspicious_ids, min_intermediaries=3):
    """Scatter-Gather: A → [B,C,D] → E (smurfing)."""
    print(f"\n  [Scatter-Gather] Detecting...")
    query = """
        UNWIND $ids AS sid
        MATCH (src:Account {id: sid})-[t1:TRANSFER]->(mid:Account)-[t2:TRANSFER]->(dst:Account)
        WHERE src <> dst AND src <> mid AND mid <> dst
        WITH src, dst, collect(DISTINCT mid.id) AS intermediaries,
             count(DISTINCT mid) AS num_mid,
             sum(t1.amount) AS scatter_amount, sum(t2.amount) AS gather_amount
        WHERE num_mid >= $minMid
        RETURN src.id AS source, dst.id AS destination,
               num_mid, intermediaries, scatter_amount, gather_amount
        ORDER BY num_mid DESC LIMIT 200
    """
    results = conn.run_query(query, {"ids": suspicious_ids, "minMid": min_intermediaries})
    print(f"    Found {len(results)} scatter-gather patterns")
    return results

def detect_structuring(conn, suspicious_ids, max_amount=10000, min_tx=5):
    """
    Structuring (Smurfing): tài khoản chia nhỏ nhiều giao dịch
    để tránh ngưỡng báo cáo. Dấu hiệu: gửi >= min_tx giao dịch
    với amount nhỏ hơn threshold trong 1 ngày.
    """
    print(f"\n  [Structuring] Detecting (amount<{max_amount:,}, min_tx={min_tx})...")
    query = """
        UNWIND $ids AS sid
        MATCH (a:Account {id: sid})-[t:TRANSFER]->(b:Account)
        WHERE t.amount < $maxAmount
        WITH a, count(t) AS small_tx_count,
             sum(t.amount) AS total_small,
             avg(t.amount) AS avg_amount,
             collect(DISTINCT b.id)[..5] AS sample_targets
        WHERE small_tx_count >= $minTx
        RETURN a.id AS account_id,
               small_tx_count, total_small, avg_amount, sample_targets
        ORDER BY small_tx_count DESC LIMIT 300
    """
    results = conn.run_query(query, {
        "ids": suspicious_ids,
        "maxAmount": max_amount,
        "minTx": min_tx
    })
    print(f"    Found {len(results)} structuring patterns")
    return results


def detect_uturn(conn, suspicious_ids):
    """
    U-turn (Round-trip): A → B → A — tiền quay ngược trở lại
    trong vòng 2 hops. Dấu hiệu của giao dịch giả tạo 2 chiều.
    """
    print(f"\n  [U-turn] Detecting (A→B→A)...")
    query = """
        UNWIND $ids AS sid
        MATCH (a:Account {id: sid})-[t1:TRANSFER]->(b:Account)-[t2:TRANSFER]->(a)
        WHERE a <> b
        WITH a, b,
             count(t1) AS fwd_count, sum(t1.amount) AS fwd_amount,
             count(t2) AS bwd_count, sum(t2.amount) AS bwd_amount
        RETURN a.id AS account_a, b.id AS account_b,
               fwd_count, fwd_amount, bwd_count, bwd_amount,
               abs(fwd_amount - bwd_amount) AS amount_diff
        ORDER BY amount_diff ASC LIMIT 300
    """
    results = conn.run_query(query, {"ids": suspicious_ids})
    print(f"    Found {len(results)} u-turn patterns")
    return results


def detect_layering(conn, suspicious_ids, min_hops=3):
    """
    Layering (Pass-through): tài khoản nhận tiền rồi chuyển ngay
    cho người khác — tỉ lệ in/out gần 1:1, nhiều hops.
    Nhẹ hơn Chain vì không traverse full path mà chỉ xem aggregation.
    """
    print(f"\n  [Layering] Detecting (pass-through accounts)...")
    query = """
        UNWIND $ids AS sid
        MATCH (a:Account {id: sid})
        WHERE a.in_degree IS NOT NULL AND a.out_degree IS NOT NULL
          AND a.in_degree >= $minHops AND a.out_degree >= $minHops
          AND a.total_sent IS NOT NULL AND a.total_received IS NOT NULL
          AND a.total_received > 0
        WITH a,
             toFloat(a.total_sent) / toFloat(a.total_received) AS flow_ratio,
             a.in_degree AS in_deg,
             a.out_degree AS out_deg
        WHERE flow_ratio >= 0.7 AND flow_ratio <= 1.3
        RETURN a.id AS account_id,
               in_deg, out_deg, flow_ratio,
               a.total_sent AS total_sent,
               a.total_received AS total_received
        ORDER BY in_deg DESC LIMIT 300
    """
    results = conn.run_query(query, {"ids": suspicious_ids, "minHops": min_hops})
    print(f"    Found {len(results)} layering (pass-through) patterns")
    return results



def save_reports(df, has_gnn=False, motifs_results=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sus = df[df["is_suspicious"]].sort_values("suspicious_score", ascending=False)
    cols = ["id", "suspicious_score", "risk_level", "criteria_met",
            "fraud_probability", "trust_score", "prediction"]
    # Thêm cột GNN nếu có
    if has_gnn:
        for c in ["gnn_fraud_prob", "gnn_prediction"]:
            if c in sus.columns:
                cols.append(c)
    for c in ["trust_level", "credit_score", "anti_trustrank_norm", "label"]:
        if c in sus.columns:
            cols.append(c)
    available = [c for c in cols if c in sus.columns]
    sus[available].to_csv(RESULTS_DIR / "suspicious_nodes.csv", index=False)
    print(f"\n  Saved: suspicious_nodes.csv ({len(sus):,} rows)")

    full_cols = ["id", "suspicious_score", "risk_level", "fraud_probability", "trust_score"]
    if has_gnn:
        full_cols.append("gnn_fraud_prob")
    available_full = [c for c in full_cols if c in df.columns]
    df[available_full].to_csv(RESULTS_DIR / "all_scores.csv", index=False)
    print(f"  Saved: all_scores.csv ({len(df):,} rows)")

    if motifs_results:
        for motif_name, records in motifs_results.items():
            if records:
                pd.DataFrame(records).to_csv(RESULTS_DIR / f"motif_{motif_name}.csv", index=False)
                print(f"  Saved: motif_{motif_name}.csv ({len(records)} patterns)")

    print(f"\n  All reports → {RESULTS_DIR}")


# MAIN PIPELINE

def run_pipeline(use_neo4j=False):
    print("\n" + "=" * 60)
    print("GRAPH MINING PIPELINE — v2.0 (Boosting + GNN + Trust)")
    print("=" * 60)

    df, has_gnn = load_data()
    df = compute_suspicious_score(df, has_gnn=has_gnn)
    df = identify_suspicious_nodes(df, has_gnn=has_gnn)

    suspicious_ids = df[df["is_suspicious"]]["id"].tolist()
    motifs_results = {}

    if use_neo4j and suspicious_ids:
        print("\n" + "=" * 60)
        print(f"PHASE 2: AML MOTIF DETECTION ({len(suspicious_ids):,} suspicious nodes)")
        print("=" * 60)
        try:
            conn = _get_neo4j_conn()
            # Rút gọn số node query motif (đồ thị sâu rất tốn thời gian tính toán)
            batch_ids = suspicious_ids[:300]
            print(f"  Processing top {len(batch_ids)} suspicious nodes...")


            motifs_results["fanout"] = detect_fanout(conn, batch_ids)
            motifs_results["fanin"] = detect_fanin(conn, batch_ids)
            motifs_results["cycle"] = detect_cycles(conn, batch_ids)
            motifs_results["scatter_gather"] = detect_scatter_gather(conn, batch_ids)
            motifs_results["structuring"] = detect_structuring(conn, batch_ids)
            motifs_results["uturn"] = detect_uturn(conn, batch_ids)
            motifs_results["layering"] = detect_layering(conn, batch_ids)

            conn.close()

            print("\n" + "-" * 60)
            print("AML MOTIF SUMMARY")
            total_patterns = 0
            for name, records in motifs_results.items():
                n = len(records)
                total_patterns += n
                print(f"  {name:<20}: {n:>6} patterns")
            print(f"  {'TOTAL':<20}: {total_patterns:>6} patterns")
        except Exception as e:
            print(f"\n  [WARNING] Neo4j error: {e}")
            print("  Skipping Phase 2.")

    save_reports(df, has_gnn=has_gnn, motifs_results=motifs_results if motifs_results else None)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return df


if __name__ == "__main__":
    df = run_pipeline(use_neo4j=True)

    sus = df[df["is_suspicious"]].head(20)
    print("\n── TOP 20 SUSPICIOUS NODES ──")
    show_cols = ["id", "suspicious_score", "risk_level", "fraud_probability", "trust_score"]
    if "gnn_fraud_prob" in df.columns:
        show_cols.insert(4, "gnn_fraud_prob")
    available = [c for c in show_cols if c in sus.columns]
    print(sus[available].to_string(index=False))
