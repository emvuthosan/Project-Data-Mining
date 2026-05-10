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

# Config
WEIGHTS = {
    "boosting": 0.55,
    "trust": 0.45,
}

SUSPICIOUS_THRESHOLD = 0.55
MIN_CRITERIA = 2


# PHASE 1: XÁC ĐỊNH SUSPICIOUS NODES (CSV only)

def load_data():
    """Load và merge dữ liệu từ Boosting + Trust Score."""
    print("\n" + "=" * 60)
    print("PHASE 1: LOAD & MERGE DATA")
    print("=" * 60)

    if not BOOSTING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {BOOSTING_CSV}")
    df_boost = pd.read_csv(BOOSTING_CSV)
    df_boost = df_boost.rename(columns={"account_id": "id"})
    print(f"[Boosting]    {len(df_boost):,} accounts loaded")

    if not TRUST_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {TRUST_CSV}")
    df_trust = pd.read_csv(TRUST_CSV)
    print(f"[Trust Score] {len(df_trust):,} accounts loaded")

    df = df_trust.merge(df_boost[["id", "fraud_probability", "prediction"]],
                        on="id", how="outer")

    df["fraud_probability"] = df["fraud_probability"].fillna(0.0)
    df["trust_score"] = df["trust_score"].fillna(0.5)
    df["prediction"] = df["prediction"].fillna(0).astype(int)

    print(f"\n[Merged]      {len(df):,} accounts total")
    print(f"  Có fraud_probability: {(df['fraud_probability'] > 0).sum():,}")
    print(f"  Có trust_score:       {(df['trust_score'] > 0).sum():,}")

    return df


def compute_suspicious_score(df):
    """
    Tính suspicious_score với xử lý riêng cho accounts có/không có trust data.

    - Có trust data (trust_score > 0): dùng blended score
    - Không có trust data (trust_score = 0): chỉ dùng fraud_probability
    """
    print("\n" + "-" * 60)
    print("COMPUTING SUSPICIOUS SCORE")
    print("-" * 60)

    has_trust = df["trust_score"] > 0
    n_with = has_trust.sum()
    n_without = (~has_trust).sum()
    print(f"  Accounts có trust data:    {n_with:,}")
    print(f"  Accounts không trust data: {n_without:,}")

    # Accounts CÓ trust data → blended score
    w = WEIGHTS
    df.loc[has_trust, "suspicious_score"] = (
        w["boosting"] * df.loc[has_trust, "fraud_probability"]
        + w["trust"] * (1 - df.loc[has_trust, "trust_score"])
    )

    # Accounts KHÔNG CÓ trust data → chỉ dùng fraud_probability
    # (không cộng thêm trust penalty vì không có dữ liệu)
    df.loc[~has_trust, "suspicious_score"] = df.loc[~has_trust, "fraud_probability"]

    print(f"  Weights (with trust): boost={w['boosting']}, trust={w['trust']}")
    print(f"  Weights (no trust):   boost=1.0 (fraud_probability only)")

    print(f"\n  Distribution:")
    print(f"    Min={df['suspicious_score'].min():.4f}  "
          f"Median={df['suspicious_score'].median():.4f}  "
          f"Max={df['suspicious_score'].max():.4f}")

    return df


def identify_suspicious_nodes(df):
    """
    Lọc node nghi ngờ dựa trên nhiều tiêu chí.

    - Có trust data: thỏa >= 2 tiêu chí
    - Không có trust data: fraud_probability >= 0.5 (Boosting tự tin)
    """
    print("\n" + "-" * 60)
    print("IDENTIFYING SUSPICIOUS NODES")
    print("-" * 60)

    has_trust = df["trust_score"] > 0

    # ── Tiêu chí cho accounts CÓ trust data ───────────────────
    c1 = df["suspicious_score"] >= SUSPICIOUS_THRESHOLD
    c2 = df["prediction"] == 1
    c3 = df["trust_level"] == "VERY_LOW" if "trust_level" in df.columns else pd.Series(False, index=df.index)
    c4 = df["anti_trustrank_norm"] >= 0.7 if "anti_trustrank_norm" in df.columns else pd.Series(False, index=df.index)

    df["criteria_met"] = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)

    # Có trust data: cần >= MIN_CRITERIA
    # Không trust data: cần fraud_probability >= 0.5 VÀ prediction == 1
    df["is_suspicious"] = False
    df.loc[has_trust, "is_suspicious"] = df.loc[has_trust, "criteria_met"] >= MIN_CRITERIA
    df.loc[~has_trust, "is_suspicious"] = (
        (df.loc[~has_trust, "fraud_probability"] >= 0.5)
        & (df.loc[~has_trust, "prediction"] == 1)
    )

    def risk_level(row):
        if row["criteria_met"] >= 4: return "CRITICAL"
        elif row["criteria_met"] >= 3: return "HIGH"
        elif row["criteria_met"] >= 2: return "MEDIUM"
        else: return "LOW"

    df["risk_level"] = df.apply(risk_level, axis=1)

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

# REPORT

def save_reports(df, motifs_results=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sus = df[df["is_suspicious"]].sort_values("suspicious_score", ascending=False)
    cols = ["id", "suspicious_score", "risk_level", "criteria_met",
            "fraud_probability", "trust_score", "prediction"]
    for c in ["trust_level", "credit_score", "anti_trustrank_norm", "label"]:
        if c in sus.columns:
            cols.append(c)
    available = [c for c in cols if c in sus.columns]
    sus[available].to_csv(RESULTS_DIR / "suspicious_nodes.csv", index=False)
    print(f"\n  Saved: suspicious_nodes.csv ({len(sus):,} rows)")

    full_cols = ["id", "suspicious_score", "risk_level", "fraud_probability", "trust_score"]
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
    print("GRAPH MINING PIPELINE")
    print("=" * 60)

    df = load_data()
    df = compute_suspicious_score(df)
    df = identify_suspicious_nodes(df)

    suspicious_ids = df[df["is_suspicious"]]["id"].tolist()
    motifs_results = {}

    if use_neo4j and suspicious_ids:
        print("\n" + "=" * 60)
        print(f"PHASE 2: AML MOTIF DETECTION ({len(suspicious_ids):,} suspicious nodes)")
        print("=" * 60)
        try:
            conn = _get_neo4j_conn()
            # Rút gọn số node query motif (đồ thị sâu rất tốn thời gian tính toán)
            batch_ids = suspicious_ids[:200]
            print(f"  Processing top {len(batch_ids)} suspicious nodes to avoid DB timeouts...")

            motifs_results["fanout"] = detect_fanout(conn, batch_ids)
            motifs_results["fanin"] = detect_fanin(conn, batch_ids)
            motifs_results["cycle"] = detect_cycles(conn, batch_ids)
            motifs_results["chain"] = detect_chains(conn, batch_ids)
            motifs_results["scatter_gather"] = detect_scatter_gather(conn, batch_ids)

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

    save_reports(df, motifs_results if motifs_results else None)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return df


if __name__ == "__main__":
    df = run_pipeline(use_neo4j=True)

    sus = df[df["is_suspicious"]].head(20)
    print("\n── TOP 20 SUSPICIOUS NODES ──")
    show_cols = ["id", "suspicious_score", "risk_level", "fraud_probability", "trust_score"]
    available = [c for c in show_cols if c in sus.columns]
    print(sus[available].to_string(index=False))
