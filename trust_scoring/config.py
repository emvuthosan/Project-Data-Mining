import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Tên GDS projection dùng chung cho tất cả thuật toán
GRAPH_NAME = "aml_graph"

# Tên node label và relationship trong Neo4j
NODE_LABEL         = "Account"
RELATIONSHIP_TYPE  = "TRANSFER"

# Tham số thuật toán
PAGERANK_CONFIG = {
    "maxIterations": 20,
    "dampingFactor": 0.85,
    "tolerance": 1e-7,
}

TRUSTRANK_CONFIG = {
    "maxIterations": 20,
    "dampingFactor": 0.85,
    "tolerance": 1e-7,
    "seed_label": 0,          # label=0 → normal account (seed tốt)
    "max_seed_count": 500,    # giới hạn số lượng seed
}

ANTI_TRUSTRANK_CONFIG = {
    "maxIterations": 20,
    "dampingFactor": 0.85,
    "tolerance": 1e-7,
    "seed_label": 1,          # label=1 → fraud account (seed xấu)
    "max_seed_count": 500,
}

SIMRANK_CONFIG = {
    "decay": 0.8,             # hệ số suy giảm C
    "maxIterations": 5,       # SimRank hội tụ nhanh hơn PageRank
    "top_k": 10,              # lấy top-k similar nodes cho mỗi node
}

TRUST_SCORE_WEIGHTS = {
    "pagerank_w":       0.20,   # trọng số PageRank
    "trustrank_w":      0.45,   # trọng số TrustRank (quan trọng nhất)
    "anti_trustrank_w": 0.30,   # trọng số Anti-TrustRank (âm)
    "simrank_w":        0.05,   # trọng số SimRank bonus
}


# Mapping trust_score → credit_score (300–900)
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 900