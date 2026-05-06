import pandas as pd
import json
import glob
import os
import joblib
import numpy as np


def _parse_rel(json_str):
    """Parse JSON string from rels CSV into a tuple for speed."""
    try:
        props = json.loads(json_str)['properties']
        return (
            props.get('amount', 0),
            props.get('is_laundering', 0),
            props.get('payment_format', ''),
            props.get('timestamp', '')
        )
    except Exception:
        return (0, 0, '', '')


def build_rels_features():
    """
    Khai thác dữ liệu giao dịch (rels_*.csv) để tạo các features cấp tài khoản.
    
    Features tạo ra (theo vai trò GỬI tiền - sender):
      - rels_tx_count: Tổng số giao dịch gửi đi
      - rels_max_amount: Giao dịch lớn nhất
      - rels_min_amount: Giao dịch nhỏ nhất
      - rels_mean_amount: Số tiền giao dịch trung bình
      - rels_std_amount: Độ lệch chuẩn số tiền giao dịch
      - rels_amount_range: Chênh lệch giữa max và min
      - rels_night_ratio: Tỷ lệ giao dịch vào ban đêm (0h-6h)
      - rels_self_tx_ratio: Tỷ lệ giao dịch tự chuyển cho chính mình
    
    Features tạo ra (theo vai trò NHẬN tiền - receiver):
      - rels_recv_count: Tổng số giao dịch nhận
      - rels_recv_mean_amount: Số tiền nhận trung bình
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    cache_file = os.path.join(data_dir, "cached_rels_features.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached rels features from {cache_file}...")
        return joblib.load(cache_file)

    rels_dir = os.path.join(data_dir, "rels")
    rels_files = sorted(glob.glob(os.path.join(rels_dir, "rels_*.csv")))

    if not rels_files:
        print("[WARNING] Không tìm thấy file rels_*.csv. Bỏ qua transaction features.")
        return None

    total = len(rels_files)
    print(f"Đang xử lý {total} file rels để trích xuất transaction features...")
    print("Lần chạy đầu sẽ mất khoảng 15-30 phút. Kết quả sẽ được cache lại.")

    # Dùng partial aggregation: mỗi file tính aggregate riêng,
    # cuối cùng gộp lại bằng các phép toán sum-compatible
    sender_aggs = []
    receiver_aggs = []

    for i, file in enumerate(rels_files):
        if (i + 1) % 20 == 0 or i == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {os.path.basename(file)}")

        raw = pd.read_csv(file)

        # Parse JSON column
        parsed = raw['r'].apply(_parse_rel)
        tx_df = pd.DataFrame(
            parsed.tolist(),
            columns=['amount', 'is_laundering', 'payment_format', 'timestamp']
        )
        tx_df['source'] = raw['source']
        tx_df['target'] = raw['target']
        tx_df['is_self_tx'] = (raw['source'] == raw['target']).astype(int)

        # Parse hour from timestamp
        tx_df['hour'] = pd.to_datetime(tx_df['timestamp'], errors='coerce').dt.hour
        tx_df['is_night'] = ((tx_df['hour'] >= 0) & (tx_df['hour'] < 6)).astype(int)

        # ===== Aggregate theo SENDER (source) =====
        s_agg = tx_df.groupby('source').agg(
            _count=('amount', 'count'),
            _sum_amount=('amount', 'sum'),
            _sum_amount_sq=('amount', lambda x: (x ** 2).sum()),
            _max_amount=('amount', 'max'),
            _min_amount=('amount', 'min'),
            _night_count=('is_night', 'sum'),
            _self_tx_count=('is_self_tx', 'sum'),
        )
        sender_aggs.append(s_agg)

        # ===== Aggregate theo RECEIVER (target) =====
        r_agg = tx_df.groupby('target').agg(
            _recv_count=('amount', 'count'),
            _recv_sum_amount=('amount', 'sum'),
        )
        receiver_aggs.append(r_agg)

        # Giải phóng bộ nhớ
        del raw, parsed, tx_df

    # ===== Gộp kết quả từ tất cả file =====
    print("Đang gộp kết quả từ tất cả các file...")

    # Sender
    sender_combined = pd.concat(sender_aggs)
    sender_final = sender_combined.groupby(level=0).agg({
        '_count': 'sum',
        '_sum_amount': 'sum',
        '_sum_amount_sq': 'sum',
        '_max_amount': 'max',
        '_min_amount': 'min',
        '_night_count': 'sum',
        '_self_tx_count': 'sum',
    })

    # Receiver
    receiver_combined = pd.concat(receiver_aggs)
    receiver_final = receiver_combined.groupby(level=0).agg({
        '_recv_count': 'sum',
        '_recv_sum_amount': 'sum',
    })

    # ===== Tính các features cuối cùng (SENDER) =====
    result = pd.DataFrame(index=sender_final.index)
    n = sender_final['_count']

    result['rels_tx_count'] = n
    result['rels_max_amount'] = sender_final['_max_amount']
    result['rels_min_amount'] = sender_final['_min_amount']
    result['rels_mean_amount'] = sender_final['_sum_amount'] / (n + 1)
    result['rels_std_amount'] = np.sqrt(
        np.maximum(
            sender_final['_sum_amount_sq'] / (n + 1)
            - (sender_final['_sum_amount'] / (n + 1)) ** 2,
            0
        )
    )
    result['rels_amount_range'] = sender_final['_max_amount'] - sender_final['_min_amount']
    result['rels_night_ratio'] = sender_final['_night_count'] / (n + 1)
    result['rels_self_tx_ratio'] = sender_final['_self_tx_count'] / (n + 1)

    result = result.reset_index().rename(columns={'source': 'node_id'})

    # ===== Tính các features cuối cùng (RECEIVER) =====
    recv_result = pd.DataFrame(index=receiver_final.index)
    rn = receiver_final['_recv_count']

    recv_result['rels_recv_count'] = rn
    recv_result['rels_recv_mean_amount'] = receiver_final['_recv_sum_amount'] / (rn + 1)

    recv_result = recv_result.reset_index().rename(columns={'target': 'node_id'})

    # ===== Merge sender + receiver =====
    result = result.merge(recv_result, on='node_id', how='outer')
    result = result.fillna(0)

    print(f"Đã tạo {len(result.columns) - 1} features từ {len(result)} tài khoản.")
    print(f"Saving to {cache_file}...")
    joblib.dump(result, cache_file)

    return result
