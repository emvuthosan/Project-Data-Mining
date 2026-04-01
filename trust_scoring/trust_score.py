import pandas as pd
import numpy as np
import config
from neo4j_connector import Neo4jConnector


# ── Absolute thresholds cho trust_level ────────────────────────────
# Sau log-transform + normalize về [0,1]:
#   VERY_LOW : < 0.30  → tài khoản hoạt động yếu, ít kết nối
#   LOW      : 0.30–0.50 → tài khoản bình thường mức thấp
#   MEDIUM   : 0.50–0.75 → tài khoản tốt
#   HIGH     : >= 0.75  → tài khoản uy tín cao
THRESHOLD_VERY_LOW = 0.30
THRESHOLD_LOW      = 0.50
THRESHOLD_MEDIUM   = 0.75


def _log_normalize(series: pd.Series) -> pd.Series:
    """
    Log-transform + min-max normalize về [0, 1].

    Lý do dùng log:
      PageRank/TrustRank có phân phối power-law: f(x) ~ x^(-a)
      Log-transform đổi thành xấp xỉ phân phối đều/Gaussian
      → normalize sẽ trải đều hơn thay vì dồn 99% nodes vào [0, 0.05]

    eps = 1e-9 để tránh log(0)
    """
    eps = 1e-9
    log_series = np.log(series + eps)
    lo, hi = log_series.min(), log_series.max()
    if hi > lo:
        return (log_series - lo) / (hi - lo)
    return pd.Series(0.5, index=series.index)


def _detect_fraud_data(df_anti_trustrank: pd.DataFrame) -> bool:
    """
    Kiểm tra xem có fraud seed thực sự không.
    Nếu toàn bộ anti_trustrank_norm = 0 → không có fraud data.
    """
    if "anti_trustrank_norm" not in df_anti_trustrank.columns:
        return False
    max_atr = df_anti_trustrank["anti_trustrank_norm"].max()
    return max_atr > 0.01  # threshold nhỏ để bỏ qua floating point noise


def compute_trust_score(
    df_pagerank: pd.DataFrame,
    df_trustrank: pd.DataFrame,
    df_anti_trustrank: pd.DataFrame,
    df_simrank: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Tính trust_score tổng hợp với log-transform và absolute thresholds.

    Thay đổi so với phiên bản cũ:
      - Normalize dùng log-transform thay vì min-max thẳng
      - Threshold cố định (0.30/0.50/0.75) thay vì percentile
      - Tự động điều chỉnh weights khi không có fraud data
    """
    weights = weights or config.TRUST_SCORE_WEIGHTS

    # ── Bước 1: Merge ──────────────────────────────────────────────
    df = df_trustrank[["id", "trustrank", "label"]].copy()

    df = df.merge(df_pagerank[["id", "pagerank"]],             on="id", how="left")
    df = df.merge(df_anti_trustrank[["id", "anti_trustrank"]], on="id", how="left")
    df = df.merge(df_simrank[["id", "avg_similarity"]],        on="id", how="left")

    # Fill missing
    for col in ["trustrank", "pagerank", "anti_trustrank", "avg_similarity"]:
        df[col] = df[col].fillna(0.0)

    # ── Bước 2: Log-transform + normalize từng score ───────────────
    # QUAN TRỌNG: dùng log-transform trước, KHÔNG dùng min-max thẳng
    # vì các score PageRank/TrustRank có phân phối power-law

    df["trustrank_norm"]      = _log_normalize(df["trustrank"])
    df["pagerank_norm"]       = _log_normalize(df["pagerank"])
    df["anti_trustrank_norm"] = _log_normalize(df["anti_trustrank"])

    # SimRank đã là structural feature, không cần log-transform
    sr_max = df["avg_similarity"].max()
    df["simrank_norm"] = df["avg_similarity"] / sr_max if sr_max > 0 else 0.5

    # ── Bước 3: Auto-detect fraud data & điều chỉnh weights ────────
    has_fraud_data = _detect_fraud_data(df)

    if not has_fraud_data:
        print("\n[Trust Score] PHÁT HIỆN: Không có fraud data (tất cả label=0).")
        print("[Trust Score] Điều chỉnh weights: bỏ Anti-TrustRank penalty.")
        # Khi không có fraud seed, anti_trustrank_norm vô nghĩa → bỏ penalty
        # Phân bổ lại weight sang TrustRank và PageRank
        w_tr  = 0.55   # tăng TrustRank
        w_pr  = 0.35   # tăng PageRank
        w_atr = 0.00   # bỏ penalty
        w_sr  = 0.10   # tăng SimRank
    else:
        w_tr  = weights["trustrank_w"]
        w_pr  = weights["pagerank_w"]
        w_atr = weights["anti_trustrank_w"]
        w_sr  = weights["simrank_w"]

    print(f"[Trust Score] Weights: TR={w_tr}, PR={w_pr}, ATR={w_atr}, SR={w_sr}")

    # ── Bước 4: Tính trust_score ────────────────────────────────────
    df["trust_score_raw"] = (
          w_tr  * df["trustrank_norm"]
        + w_pr  * df["pagerank_norm"]
        - w_atr * df["anti_trustrank_norm"]
        + w_sr  * df["simrank_norm"]
    )

    # Clip về [0, 1]
    df["trust_score"] = df["trust_score_raw"].clip(0.0, 1.0)

    # Normalize lại về đúng [0, 1] sau clip
    ts_min, ts_max = df["trust_score"].min(), df["trust_score"].max()
    if ts_max > ts_min:
        df["trust_score"] = (df["trust_score"] - ts_min) / (ts_max - ts_min)

    # ── Bước 5: Map → credit_score [300, 900] ──────────────────────
    cs_min = config.CREDIT_SCORE_MIN  # 300
    cs_max = config.CREDIT_SCORE_MAX  # 900
    df["credit_score"] = (
        cs_min + df["trust_score"] * (cs_max - cs_min)
    ).round(0).astype(int)

    # ── Bước 6: Phân loại trust_level (ABSOLUTE thresholds) ────────
    # Dùng ngưỡng cố định thay vì percentile
    # Lý do: percentile-based sẽ luôn cho 50% VERY_LOW bất kể dữ liệu thế nào
    def classify_trust(score: float) -> str:
        if score >= THRESHOLD_MEDIUM:
            return "HIGH"
        elif score >= THRESHOLD_LOW:
            return "MEDIUM"
        elif score >= THRESHOLD_VERY_LOW:
            return "LOW"
        else:
            return "VERY_LOW"

    df["trust_level"] = df["trust_score"].apply(classify_trust)

    # ── Bước 7: Phân loại credit_level ─────────────────────────────
    def classify_credit(cs: int) -> str:
        if cs >= 750:   return "Excellent"
        elif cs >= 650: return "Good"
        elif cs >= 550: return "Fair"
        elif cs >= 450: return "Poor"
        else:           return "Very Poor"

    df["credit_level"] = df["credit_score"].apply(classify_credit)

    df = df.sort_values("credit_score", ascending=False).reset_index(drop=True)

    # ── In thống kê ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRUST SCORE SUMMARY")
    print("="*60)
    print(f"Tổng số accounts: {len(df):,}")
    print(f"Trust score: [{df['trust_score'].min():.4f}, {df['trust_score'].max():.4f}]")
    print(f"Credit score: [{df['credit_score'].min()}, {df['credit_score'].max()}]")

    print(f"\nThresholds (absolute): "
          f"VERY_LOW < {THRESHOLD_VERY_LOW} | "
          f"LOW {THRESHOLD_VERY_LOW}-{THRESHOLD_LOW} | "
          f"MEDIUM {THRESHOLD_LOW}-{THRESHOLD_MEDIUM} | "
          f"HIGH >= {THRESHOLD_MEDIUM}")

    print("\nPhân bố trust_level:")
    level_counts = df["trust_level"].value_counts()
    total = len(df)
    for level in ["HIGH", "MEDIUM", "LOW", "VERY_LOW"]:
        n = level_counts.get(level, 0)
        pct = n / total * 100
        print(f"  {level:<10}: {n:>10,} ({pct:.1f}%)")

    print("\nPhân bố credit_level:")
    credit_counts = df["credit_level"].value_counts()
    for level in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]:
        n = credit_counts.get(level, 0)
        pct = n / total * 100
        print(f"  {level:<12}: {n:>10,} ({pct:.1f}%)")

    # Debug: kiểm tra log-norm distribution
    print(f"\nKiểm tra trustrank_norm sau log-transform:")
    print(f"  Mean={df['trustrank_norm'].mean():.4f} | "
          f"Std={df['trustrank_norm'].std():.4f} | "
          f"P25={df['trustrank_norm'].quantile(0.25):.4f} | "
          f"P50={df['trustrank_norm'].quantile(0.50):.4f} | "
          f"P75={df['trustrank_norm'].quantile(0.75):.4f}")
    print("="*60)

    return df


def write_trust_scores_to_neo4j(conn: Neo4jConnector, df: pd.DataFrame):
    """
    Ghi trust_score, credit_score, trust_level ngược lại vào Neo4j.
    """
    print("[Neo4j Write] Đang ghi trust scores vào Neo4j...")

    BATCH_SIZE = 500
    records = df[["id", "trust_score", "credit_score", "trust_level", "credit_level"]].to_dict("records")

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        write_query = """
            UNWIND $batch AS row
            MATCH (a:Account {id: row.id})
            SET a.trust_score   = row.trust_score,
                a.credit_score  = row.credit_score,
                a.trust_level   = row.trust_level,
                a.credit_level  = row.credit_level
        """
        conn.run_write(write_query, {"batch": batch})
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"[Neo4j Write] Đã ghi {i + len(batch):,} / {len(records):,}")

    print(f"[Neo4j Write] Hoàn thành. Đã ghi {len(df):,} accounts.")