import time
import pandas as pd
import config
from neo4j_connector import Neo4jConnector

SEED_LIMIT = 200   # giới hạn fraud seeds (đừng tăng > 300 trên máy 8GB)


def run_anti_trustrank(conn: Neo4jConnector,
                        graph_name: str = config.GRAPH_NAME,
                        cfg: dict = None) -> pd.DataFrame:
    """
    Chạy Anti-TrustRank (Personalized PageRank từ fraud seed nodes).
    Phiên bản tối ưu memory cho máy 8GB RAM.

    Returns:
        DataFrame với cột [id, anti_trustrank, anti_trustrank_norm, label]
    """
    cfg = cfg or config.ANTI_TRUSTRANK_CONFIG
    print("[Anti-TrustRank] Đang chuẩn bị fraud seed nodes...")

    # ── Bước 1: Lấy fraud seed nodes (label=1) ─────────────────────
    seed_query = """
        MATCH (a:Account {label: $label})
        RETURN a.id AS id
        LIMIT $max_count
    """
    records = conn.run(seed_query, {
        "label":     cfg["seed_label"],
        "max_count": SEED_LIMIT          # dùng hằng số cứng, không dùng cfg
    })
    seed_ids = [r["id"] for r in records]

    # ── Bước 2: Trường hợp không có fraud seed ─────────────────────
    # KHÔNG gọi "MATCH (a:Account) SET a.anti_trustrank = 0.0"
    # vì đó là full-scan write → OOM
    # Thay bằng: trả DataFrame zeros hoàn toàn trong Python
    if not seed_ids:
        print("[Anti-TrustRank] Không có fraud seed → gán anti_trustrank=0 (Python only, không ghi Neo4j).")
        all_nodes_query = """
            MATCH (a:Account)
            RETURN a.id AS id, a.label AS label
        """
        node_records = conn.run(all_nodes_query)
        df = pd.DataFrame(node_records)
        df["anti_trustrank"]      = 0.0
        df["anti_trustrank_norm"] = 0.0
        return df

    print(f"[Anti-TrustRank] {len(seed_ids)} fraud seed nodes")

    # ── Bước 3: Chạy Personalized PageRank ─────────────────────────
    # KHÔNG drop + re-project → dùng lại projection đã tạo ở pipeline Bước 1
    # KHÔNG gán property lên nodes trước khi gọi GDS
    print("[Anti-TrustRank] Đang chạy PPR từ fraud seeds...")

    ppr_query = """
        MATCH (a:Account)
        WHERE a.id IN $ids
        WITH collect(a) AS fraudSeeds
        CALL gds.pageRank.write(
            $graphName,
            {
                maxIterations:  $maxIterations,
                dampingFactor:  $dampingFactor,
                tolerance:      $tolerance,
                sourceNodes:    fraudSeeds,
                writeProperty:  'anti_trustrank'
            }
        )
        YIELD nodePropertiesWritten, ranIterations, didConverge
        RETURN nodePropertiesWritten, ranIterations, didConverge
    """
    params = {
        "graphName":     graph_name,
        "maxIterations": cfg["maxIterations"],
        "dampingFactor": cfg["dampingFactor"],
        "tolerance":     cfg["tolerance"],
        "ids":           seed_ids,
    }

    t0 = time.time()
    result = conn.run(ppr_query, params)
    elapsed = time.time() - t0

    if result:
        r = result[0]
        print(
            f"[Anti-TrustRank] Xong: {r['nodePropertiesWritten']} nodes | "
            f"Vòng: {r['ranIterations']} | Hội tụ: {r['didConverge']} | "
            f"Thời gian: {elapsed:.1f}s"
        )
    else:
        print("[Anti-TrustRank] Cảnh báo: GDS không trả về kết quả!")

    # ── Bước 4: Đọc kết quả ────────────────────────────────────────
    print("[Anti-TrustRank] Đọc kết quả từ Neo4j...")
    read_query = """
        MATCH (a:Account)
        WHERE a.anti_trustrank IS NOT NULL
        RETURN a.id     AS id,
               a.anti_trustrank AS anti_trustrank,
               a.label          AS label
        ORDER BY anti_trustrank DESC
    """
    records = conn.run(read_query)

    if not records:
        print("[Anti-TrustRank] Không đọc được anti_trustrank — gán 0 (Python only).")
        all_nodes_query = """
            MATCH (a:Account)
            RETURN a.id AS id, a.label AS label
        """
        node_records = conn.run(all_nodes_query)
        df = pd.DataFrame(node_records)
        df["anti_trustrank"]      = 0.0
        df["anti_trustrank_norm"] = 0.0
        return df

    df = pd.DataFrame(records)

    # ── Bước 5: Chuẩn hoá về [0, 1] ───────────────────────────────
    atr_min = df["anti_trustrank"].min()
    atr_max = df["anti_trustrank"].max()
    if atr_max > atr_min:
        df["anti_trustrank_norm"] = (df["anti_trustrank"] - atr_min) / (atr_max - atr_min)
    else:
        df["anti_trustrank_norm"] = 0.0

    print(f"[Anti-TrustRank] Hoàn thành. {len(df)} nodes.")
    print(df[["id", "anti_trustrank", "anti_trustrank_norm", "label"]].head(10).to_string(index=False))
    return df