import time
import pandas as pd
import numpy as np
from collections import defaultdict
import config
from neo4j_connector import Neo4jConnector

# ── Hằng số ────────────────────────────────────────────────────────
SUBGRAPH_LIMIT     = 3000   # số nodes cho SimRank Python chính xác
SUBGRAPH_EDGE_MULT = 10     # edges tải = SUBGRAPH_LIMIT × bội số này
FALLBACK_NODE_LIMIT = 100000


# ══════════════════════════════════════════════════════════════════
# CÁCH A: SimRank xấp xỉ từ structural features (< 1 phút)
# ══════════════════════════════════════════════════════════════════

def run_simrank_fast(conn: Neo4jConnector,
                     graph_name: str = config.GRAPH_NAME,
                     cfg: dict = None) -> pd.DataFrame:
    """
    SimRank xấp xỉ NHANH dựa trên structural similarity features.

    Nguyên lý:
      Thay vì so sánh từng cặp node O(N²), tính "structural similarity score"
      của mỗi node dựa trên:
        - in_degree  : số account gửi tiền đến
        - out_degree : số account nhận tiền từ
        - balance    : tỷ lệ cân bằng in/out (node bình thường thì cân bằng)
        - pagerank   : đã tính ở Bước 2, tái sử dụng không tốn thêm RAM

      simrank_norm cao = node có cấu trúc kết nối "bình thường"
      simrank_norm thấp = node cô lập hoặc có cấu trúc bất thường (đáng ngờ)

    Thời gian: < 1 phút trên 2M nodes
    RAM: chỉ đọc, không ghi thêm vào Neo4j
    """
    cfg = cfg or config.SIMRANK_CONFIG
    print("[SimRank-Fast] Tính SimRank xấp xỉ từ structural features...")
    t0 = time.time()

    # ── Đọc degree + pagerank (pagerank đã có từ Bước 2) ──────────
    degree_query = """
        MATCH (a:Account)
        OPTIONAL MATCH (a)-[:TRANSFER]->(out_node:Account)
        OPTIONAL MATCH (in_node:Account)-[:TRANSFER]->(a)
        WITH a,
             count(DISTINCT out_node) AS out_deg,
             count(DISTINCT in_node)  AS in_deg,
             coalesce(a.pagerank, 0.0) AS pr
        RETURN a.id          AS id,
               in_deg, out_deg, pr,
               coalesce(a.label, 0)  AS label
    """
    print("[SimRank-Fast] Đọc degree + pagerank từ Neo4j...")
    records = conn.run(degree_query)

    if not records:
        print("[SimRank-Fast] Không đọc được dữ liệu → fallback zeros.")
        return _empty_simrank_df(conn)

    df = pd.DataFrame(records)
    print(f"[SimRank-Fast] Đọc xong {len(df)} nodes ({time.time()-t0:.1f}s)")

    # ── Chuẩn hoá từng feature về [0, 1] ─────────────────────────
    for col in ["in_deg", "out_deg", "pr"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (
            (df[col] - col_min) / (col_max - col_min)
            if col_max > col_min else 0.0
        )

    # ── Tỷ lệ cân bằng in/out ────────────────────────────────────
    total_deg = df["in_deg"] + df["out_deg"]
    df["balance"] = np.where(
        total_deg > 0,
        1.0 - np.abs(df["in_deg"] - df["out_deg"]) / (total_deg + 1e-9),
        0.0
    )

    # ── SimRank xấp xỉ = tổng hợp có trọng số ────────────────────
    df["avg_similarity"] = (
        0.40 * df["balance"]
        + 0.30 * df["pr_norm"]
        + 0.15 * df["in_deg_norm"]
        + 0.15 * df["out_deg_norm"]
    )

    # ── Chuẩn hoá về [0, 1] ──────────────────────────────────────
    sim_max = df["avg_similarity"].max()
    df["simrank_norm"] = df["avg_similarity"] / sim_max if sim_max > 0 else 0.0

    elapsed = time.time() - t0
    print(f"[SimRank-Fast] Hoàn thành {len(df)} nodes trong {elapsed:.1f}s")
    print(f"[SimRank-Fast] simrank_norm: "
          f"min={df['simrank_norm'].min():.4f}, max={df['simrank_norm'].max():.4f}, "
          f"mean={df['simrank_norm'].mean():.4f}")

    return df[["id", "avg_similarity", "simrank_norm", "label"]]


# ══════════════════════════════════════════════════════════════════
# CÁCH B: SimRank chính xác trên subgraph nhỏ (~5 phút)
# ══════════════════════════════════════════════════════════════════

def run_simrank_subgraph(conn: Neo4jConnector,
                          graph_name: str = config.GRAPH_NAME,
                          cfg: dict = None,
                          node_limit: int = SUBGRAPH_LIMIT) -> pd.DataFrame:
    """
    Tính SimRank chính xác trên subgraph nhỏ gồm các node quan trọng nhất.

    Chiến lược:
      1. Chọn top node_limit nodes theo (anti_trustrank + pagerank) DESC
         → đây là các nodes cần phân biệt nhất
      2. Tính SimRank chính xác trên subgraph đó
      3. Merge với toàn bộ nodes, gán neutral=0.5 cho phần còn lại

    Thời gian: ~5 phút cho 3000 nodes
    """
    cfg = cfg or config.SIMRANK_CONFIG
    C        = cfg["decay"]           # 0.8
    max_iter = cfg["maxIterations"]   # 5

    print(f"[SimRank-Subgraph] Tính SimRank chính xác trên {node_limit} nodes quan trọng...")
    t0 = time.time()

    # ── Bước 1: Chọn nodes quan trọng ────────────────────────────
    select_query = """
        MATCH (a:Account)
        WITH a,
             coalesce(a.anti_trustrank, 0.0) AS atr,
             coalesce(a.pagerank, 0.0)        AS pr
        ORDER BY (atr + pr) DESC
        LIMIT $limit
        RETURN a.id AS id
    """
    selected = conn.run(select_query, {"limit": node_limit})
    selected_ids = set(r["id"] for r in selected)
    print(f"[SimRank-Subgraph] Chọn được {len(selected_ids)} nodes ({time.time()-t0:.1f}s)")

    # ── Bước 2: Tải edges trong subgraph ─────────────────────────
    edge_query = """
        MATCH (a:Account)-[:TRANSFER]->(b:Account)
        WHERE a.id IN $ids AND b.id IN $ids
        RETURN a.id AS src, b.id AS dst
        LIMIT $limit
    """
    edges = conn.run(edge_query, {
        "ids":   list(selected_ids),
        "limit": node_limit * SUBGRAPH_EDGE_MULT,
    })
    print(f"[SimRank-Subgraph] Tải {len(edges)} edges trong subgraph ({time.time()-t0:.1f}s)")

    if not edges:
        print("[SimRank-Subgraph] Subgraph rỗng → fallback Fast mode")
        return run_simrank_fast(conn, graph_name, cfg)

    # ── Bước 3: Build in-neighbor index ──────────────────────────
    nodes = set()
    in_neighbors = defaultdict(set)
    for e in edges:
        src, dst = e["src"], e["dst"]
        nodes.add(src)
        nodes.add(dst)
        in_neighbors[dst].add(src)

    node_list = sorted(list(nodes))
    node_idx  = {n: i for i, n in enumerate(node_list)}
    N         = len(node_list)
    print(f"[SimRank-Subgraph] Subgraph thực tế: {N} nodes")

    # ── Bước 4: Tính SimRank (N×N float32) ───────────────────────
    in_nb_idx = {
        node_idx[v]: [node_idx[u] for u in in_neighbors[v] if u in node_idx]
        for v in node_list
    }

    S = np.eye(N, dtype=np.float32)

    for iteration in range(max_iter):
        S_new = np.eye(N, dtype=np.float32)
        for i in range(N):
            Ii = in_nb_idx.get(i, [])
            if not Ii:
                continue
            Ii_arr = np.array(Ii, dtype=np.int32)
            for j in range(i + 1, N):
                Ij = in_nb_idx.get(j, [])
                if not Ij:
                    continue
                Ij_arr = np.array(Ij, dtype=np.int32)
                # Dùng numpy ix_ indexing: ~10x nhanh hơn vòng lặp Python
                total = float(S[np.ix_(Ii_arr, Ij_arr)].sum())
                s_ij  = C / (len(Ii) * len(Ij)) * total
                S_new[i][j] = s_ij
                S_new[j][i] = s_ij

        delta = float(np.abs(S_new - S).max())
        S = S_new
        print(f"  Iter {iteration+1}/{max_iter} | delta={delta:.6f} | {time.time()-t0:.1f}s")
        if delta < 1e-5:
            print("  Hội tụ sớm.")
            break

    # ── Bước 5: Tổng hợp avg similarity ──────────────────────────
    np.fill_diagonal(S, 0.0)
    avg_sim = (S.sum(axis=1) / max(N - 1, 1)).tolist()

    df_sub = pd.DataFrame({
        "id":    node_list,
        "avg_similarity": avg_sim,
    })

    # ── Bước 6: Merge với toàn bộ nodes ──────────────────────────
    all_nodes_query = """
        MATCH (a:Account)
        RETURN a.id          AS id,
               coalesce(a.label, 0)  AS label
    """
    all_records = conn.run(all_nodes_query)
    df_all = pd.DataFrame(all_records)

    df_result = df_all.merge(df_sub, on="id", how="left")
    # Nodes không trong subgraph → 0.5 (neutral, không biết)
    df_result["avg_similarity"] = df_result["avg_similarity"].fillna(0.5)

    sim_max = df_result["avg_similarity"].max()
    df_result["simrank_norm"] = (
        df_result["avg_similarity"] / sim_max if sim_max > 0 else 0.0
    )

    elapsed = time.time() - t0
    print(f"[SimRank-Subgraph] Hoàn thành {len(df_result)} nodes trong {elapsed:.1f}s")
    return df_result[["id", "avg_similarity", "simrank_norm", "label"]]


# ══════════════════════════════════════════════════════════════════
# Helper: fallback DataFrame rỗng
# ══════════════════════════════════════════════════════════════════

def _empty_simrank_df(conn: Neo4jConnector) -> pd.DataFrame:
    """Trả về DataFrame simrank_norm=0 khi có lỗi nghiêm trọng."""
    query = f"""
        MATCH (a:Account)
        RETURN a.id         AS id,
               coalesce(a.label, 0) AS label
        LIMIT {FALLBACK_NODE_LIMIT}
    """
    records = conn.run(query)
    df = pd.DataFrame(records)
    df["avg_similarity"] = 0.0
    df["simrank_norm"]   = 0.0
    return df


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def run_simrank(conn: Neo4jConnector,
                graph_name: str = config.GRAPH_NAME,
                cfg: dict = None,
                mode: str = "fast") -> pd.DataFrame:
    """
    Entry point cho SimRank. Chọn mode phù hợp:

    ┌──────────────┬────────────────────────────────────────┬──────────┐
    │ mode         │ Phù hợp với                            │ Thời gian│
    ├──────────────┼────────────────────────────────────────┼──────────┤
    │ "fast"       │ 2M+ nodes, máy 8GB (DEFAULT)           │ < 1 phút │
    │ "subgraph"   │ Cần chính xác hơn, vẫn nhanh           │ ~5 phút  │
    │ "gds"        │ Bị chặn — tự động chuyển sang "fast"   │ N/A      │
    └──────────────┴────────────────────────────────────────┴──────────┘

    Ghi chú: SimRank chỉ đóng góp 5% vào trust_score (w_sr=0.05)
    → mode="fast" là đủ tốt cho mục đích này.
    """
    if mode == "fast":
        return run_simrank_fast(conn, graph_name, cfg)

    elif mode == "subgraph":
        return run_simrank_subgraph(conn, graph_name, cfg)

    elif mode == "gds":
        print("━" * 55)
        print("[SimRank] CẢNH BÁO: mode='gds' với 2M nodes sẽ mất")
        print("[SimRank] hàng tiếng. Tự động chuyển sang mode='fast'.")
        print("[SimRank] Để dùng GDS thật sự: sửa dòng này trong code")
        print("[SimRank] và chỉ dùng khi graph < 100K nodes.")
        print("━" * 55)
        return run_simrank_fast(conn, graph_name, cfg)

    else:
        raise ValueError(
            f"[SimRank] mode không hợp lệ: '{mode}'. "
            f"Chọn 'fast', 'subgraph', hoặc 'gds'."
        )