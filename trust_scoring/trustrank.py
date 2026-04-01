import time
import pandas as pd
import config
from neo4j_connector import Neo4jConnector

SEED_LIMIT = 2000


def _get_quality_seeds(conn: Neo4jConnector, max_count: int) -> list:
    """
    Chọn seed chất lượng:
      - label = 0 (normal confirmed)
      - degree cao (active, không cô lập)
      - Không kề trực tiếp với fraud node (1-hop clean)

    Nếu không đủ → fallback lấy label=0 degree cao nhất
    """
    print("[TrustRank] Chọn seed nodes chất lượng cao...")

    strict_query = """
        MATCH (a:Account {label: 0})
        WHERE NOT EXISTS {
            MATCH (a)-[:TRANSFER]-(fraud:Account {label: 1})
        }
        WITH a, coalesce(a.in_degree, 0) + coalesce(a.out_degree, 0) AS deg
        ORDER BY deg DESC
        LIMIT $limit
        RETURN a.id AS id, deg
    """
    records = conn.run(strict_query, {"limit": max_count})
    ids = [r["id"] for r in records]

    if len(ids) >= max_count // 2:
        avg_deg = sum(r["deg"] for r in records) / max(len(records), 1)
        print(f"[TrustRank] Strict seed: {len(ids)} nodes sạch, avg_degree={avg_deg:.1f}")
        return ids

    print(f"[TrustRank] Strict chỉ có {len(ids)}, fallback degree cao...")
    fallback_query = """
        MATCH (a:Account {label: 0})
        WITH a, coalesce(a.in_degree, 0) + coalesce(a.out_degree, 0) AS deg
        ORDER BY deg DESC
        LIMIT $limit
        RETURN a.id AS id, deg
    """
    records = conn.run(fallback_query, {"limit": max_count})
    ids = [r["id"] for r in records]
    print(f"[TrustRank] Fallback seed: {len(ids)} nodes label=0 degree cao")
    return ids


def run_trustrank(conn: Neo4jConnector,
                  graph_name: str = config.GRAPH_NAME,
                  cfg: dict = None) -> pd.DataFrame:
    """
    Chạy TrustRank (Personalized PageRank từ seed tốt).

    Returns:
        DataFrame [id, trustrank, trustrank_norm, label]
    """
    cfg = cfg or config.TRUSTRANK_CONFIG
    print("[TrustRank] Đang chuẩn bị seed nodes...")

    seed_ids = _get_quality_seeds(conn, SEED_LIMIT)

    if not seed_ids:
        raise ValueError(
            "[TrustRank] Không tìm thấy seed nodes với label=0.\n"
            "→ Hãy chạy: python derive_labels.py trước!"
        )

    # ── Chạy PPR (không drop/re-project) ──────────────────────────
    print(f"[TrustRank] Chạy PPR với {len(seed_ids)} seed nodes...")
    ppr_query = """
        MATCH (a:Account)
        WHERE a.id IN $ids
        WITH collect(a) AS sources
        CALL gds.pageRank.write(
            $graphName,
            {
                maxIterations:  $maxIterations,
                dampingFactor:  $dampingFactor,
                tolerance:      $tolerance,
                sourceNodes:    sources,
                writeProperty:  'trustrank'
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
    if result:
        r = result[0]
        print(f"[TrustRank] Xong: {r['nodePropertiesWritten']} nodes | "
              f"Vòng: {r['ranIterations']} | Hội tụ: {r['didConverge']} | {time.time()-t0:.1f}s")

    # ── Đọc kết quả ────────────────────────────────────────────────
    read_query = """
        MATCH (a:Account)
        WHERE a.trustrank IS NOT NULL
        RETURN a.id AS id,
               a.trustrank  AS trustrank,
               a.label      AS label
        ORDER BY trustrank DESC
    """
    records = conn.run(read_query)

    if not records:
        print("[TrustRank] Không đọc được trustrank — gán 0.")
        fallback = conn.run("MATCH (a:Account) RETURN a.id AS id, a.label AS label")
        df = pd.DataFrame(fallback)
        df["trustrank"] = 0.0
        df["trustrank_norm"] = 0.0
        return df

    df = pd.DataFrame(records)

    # ── Debug: in phân bố để kiểm tra ────────────────────────────
    print(f"\n[TrustRank] Phân bố trustrank:")
    print(f"  Min={df['trustrank'].min():.6f} | "
          f"Median={df['trustrank'].median():.6f} | "
          f"Max={df['trustrank'].max():.6f} | "
          f"Std={df['trustrank'].std():.6f}")

    if "label" in df.columns:
        for lbl, name in [(0, "normal"), (1, "fraud")]:
            sub = df[df["label"] == lbl]
            if not sub.empty:
                print(f"  {name}: n={len(sub):,}, avg={sub['trustrank'].mean():.6f}")
        if not df[df["label"]==1].empty and not df[df["label"]==0].empty:
            sep = df[df["label"]==0]["trustrank"].mean() - df[df["label"]==1]["trustrank"].mean()
            print(f"  separation (normal-fraud): {sep:+.6f} {'✓' if sep > 0 else '✗'}")

    # ── Chuẩn hoá [0, 1] ─────────────────────────────────────────
    tr_min, tr_max = df["trustrank"].min(), df["trustrank"].max()
    if tr_max > tr_min:
        df["trustrank_norm"] = (df["trustrank"] - tr_min) / (tr_max - tr_min)
    else:
        df["trustrank_norm"] = 0.5

    print(f"\n[TrustRank] Hoàn thành. {len(df)} nodes.")
    return df