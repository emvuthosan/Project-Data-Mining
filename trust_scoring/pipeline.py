import time
import pandas as pd
import config
from neo4j_connector import Neo4jConnector
from graph_projection import create_projection
from pagerank import run_pagerank
from trustrank import run_trustrank
from anti_trustrank import run_anti_trustrank
from simrank import run_simrank
from trust_score import compute_trust_score, write_trust_scores_to_neo4j


def run_pipeline(
    save_csv:     bool = True,
    write_neo4j:  bool = True,
    simrank_mode: str  = "fast",
) -> pd.DataFrame:
    start_time = time.time()
    print("\n" + "="*60)
    print("TRUST SCORING PIPELINE — BẮT ĐẦU")
    print(f"SimRank mode: {simrank_mode}")
    print("="*60)

    with Neo4jConnector() as conn:

        print("\n[BƯỚC 1] Tạo GDS Graph Projection...")
        t0 = time.time()
        create_projection(conn)
        print(f"  → {time.time()-t0:.1f}s")

        print("\n[BƯỚC 2] Tính PageRank...")
        t0 = time.time()
        df_pr = run_pagerank(conn)
        print(f"  → {time.time()-t0:.1f}s")

        print("\n[BƯỚC 3] Tính TrustRank...")
        t0 = time.time()
        df_tr = run_trustrank(conn)
        print(f"  → {time.time()-t0:.1f}s")

        print("\n[BƯỚC 4] Tính Anti-TrustRank...")
        t0 = time.time()
        df_atr = run_anti_trustrank(conn)
        print(f"  → {time.time()-t0:.1f}s")

        print(f"\n[BƯỚC 5] Tính SimRank (mode='{simrank_mode}')...")
        t0 = time.time()
        df_sr = run_simrank(conn, mode=simrank_mode)
        print(f"  → {time.time()-t0:.1f}s")

        print("\n[BƯỚC 6] Tổng hợp trust_score (bao gồm behavioral features)...")
        t0 = time.time()
        # Truyền conn để trust_score tự tải behavioral features từ Neo4j
        df_final = compute_trust_score(
            df_pagerank=df_pr,
            df_trustrank=df_tr,
            df_anti_trustrank=df_atr,
            df_simrank=df_sr,
        )
        print(f"  → {time.time()-t0:.1f}s")

        if write_neo4j:
            print("\n[BƯỚC 7] Ghi kết quả vào Neo4j...")
            t0 = time.time()
            write_trust_scores_to_neo4j(conn, df_final)
            print(f"  → {time.time()-t0:.1f}s")

    if save_csv:
        output_cols = [
            "id", "trust_score", "credit_score",
            "trust_level", "credit_level",
            "trustrank_norm", "pagerank_norm",
            "anti_trustrank_norm", "simrank_norm",
            "behavioral_anomaly_norm", "label"
        ]
        available = [c for c in output_cols if c in df_final.columns]
        df_final[available].to_csv("trust_scores_output.csv", index=False)
        print("\n[Output] Đã lưu → trust_scores_output.csv")

    total = time.time() - start_time
    print("\n" + "="*60)
    print(f"PIPELINE HOÀN THÀNH ({total:.1f}s)")
    print("="*60)
    return df_final


if __name__ == "__main__":
    df = run_pipeline(
        save_csv=True,
        write_neo4j=True,
        simrank_mode="subgraph",
    )

    print("\n── TOP 20 ACCOUNTS TIN CẬY NHẤT ──")
    cols = ["id", "credit_score", "trust_level",
            "trustrank_norm", "behavioral_anomaly_norm"]
    print(df.head(20)[[c for c in cols if c in df.columns]].to_string(index=False))

    print("\n── TOP 20 ACCOUNTS NGUY HIỂM NHẤT ──")
    print(df.tail(20)[[c for c in cols if c in df.columns]].to_string(index=False))