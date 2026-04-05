import numpy as np
import pandas as pd
import config
from neo4j_connector import Neo4jConnector


def run_pagerank(conn: Neo4jConnector,
                 graph_name: str = config.GRAPH_NAME,
                 cfg: dict = None) -> pd.DataFrame:
    """
    Chạy PageRank bằng GDS và ghi kết quả vào property 'pagerank'.

    Returns:
        DataFrame với cột [id, pagerank, pagerank_norm]
        pagerank_norm = log-normalized (để debug), trust_score.py sẽ tự tính lại
    """
    cfg = cfg or config.PAGERANK_CONFIG

    print("[PageRank] Đang chạy thuật toán...")

    write_query = """
        CALL gds.pageRank.write(
            $graphName,
            {
                maxIterations:  $maxIterations,
                dampingFactor:  $dampingFactor,
                tolerance:      $tolerance,
                writeProperty:  'pagerank'
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
    }
    result = conn.run(write_query, params)
    if result:
        r = result[0]
        print(
            f"[PageRank] Viết {r['nodePropertiesWritten']} nodes | "
            f"Số vòng: {r['ranIterations']} | Hội tụ: {r['didConverge']}"
        )

    read_query = """
        MATCH (a:Account)
        RETURN a.id       AS id,
               a.pagerank AS pagerank
        ORDER BY pagerank DESC
    """
    records = conn.run(read_query)
    df = pd.DataFrame(records)

    if df.empty:
        print("[PageRank] Cảnh báo: không đọc được kết quả!")
        return df

    # Log-normalize để debug (trust_score.py sẽ tính lại từ raw)
    eps = 1e-9
    log_pr = np.log(df["pagerank"] + eps)
    lo, hi = log_pr.min(), log_pr.max()
    df["pagerank_norm"] = (log_pr - lo) / (hi - lo) if hi > lo else 0.5

    print(f"[PageRank] Hoàn thành. {len(df):,} nodes.")
    print(f"[PageRank] Raw: min={df['pagerank'].min():.2e}, "
          f"max={df['pagerank'].max():.2e}, "
          f"mean={df['pagerank'].mean():.2e}")
    print(f"[PageRank] Log-norm: min={df['pagerank_norm'].min():.4f}, "
          f"max={df['pagerank_norm'].max():.4f}, "
          f"mean={df['pagerank_norm'].mean():.4f}")
    print(df[["id", "pagerank", "pagerank_norm"]].head(10).to_string(index=False))
    return df