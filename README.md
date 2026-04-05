# Graph-based AML & Trust Scoring System

Hệ thống phân tích đồ thị phục vụ **phát hiện rửa tiền (AML)** và **chấm điểm tín nhiệm tài khoản (Trust Scoring)** dựa trên Neo4j, Graph Mining và Machine Learning.

Dự án mô hình hóa giao dịch tài chính thành đồ thị có hướng, sau đó khai thác các đặc trưng cấu trúc để:

- phát hiện tài khoản gian lận,
- truy vết các mô-típ rửa tiền,
- phân cụm cộng đồng nghi vấn,
- trực quan hóa mạng lưới giao dịch phục vụ điều tra và giám sát. :contentReference[oaicite:2]{index=2}

## Mục tiêu

- Xây dựng pipeline dữ liệu từ giao dịch thô đến đồ thị Neo4j.
- Tính toán các chỉ số đồ thị như PageRank, TrustRank, Anti-TrustRank, SimRank.
- Hợp nhất đặc trưng đồ thị và đặc trưng thống kê để huấn luyện mô hình phân loại tài khoản gian lận.
- Phát hiện các mô-típ AML như Cycle, Fan-in, Fan-out, Stack, Scatter-Gather, v.v.
- Cung cấp dashboard trực quan cho phân tích và điều tra. :contentReference[oaicite:3]{index=3}

## Kiến trúc hệ thống

Dự án được chia thành 5 phân hệ chính:

1. **Data Engineering & Neo4j Modeling**
   - EDA, làm sạch dữ liệu
   - Thiết kế node/edge schema
   - Nạp dữ liệu vào Neo4j
   - Tạo node features và index/constraint

2. **Trust Scoring**
   - Tính PageRank
   - Xây dựng TrustRank / Anti-TrustRank
   - Tính SimRank
   - Tổng hợp thành `trust_score`

3. **Machine Learning / Node Classification**
   - Kết hợp node features và graph features
   - Xử lý mất cân bằng lớp
   - Huấn luyện mô hình phân loại fraud/normal

4. **AML Graph Mining**
   - Truy vết motif rửa tiền
   - Phát hiện cộng đồng nghi vấn
   - Gắn nhãn kiểu gian lận cho từng community

5. **Dashboard & Demo Pipeline**
   - Xây dựng giao diện trực quan
   - Hiển thị trust leaderboard
   - Watchlist tài khoản nghi vấn
   - Fraud community dashboard
   - Interactive graph canvas :contentReference[oaicite:4]{index=4}

## Tính năng nổi bật

- Mô hình hóa giao dịch tài chính dưới dạng đồ thị có hướng.
- Chấm điểm tín nhiệm tài khoản bằng các thuật toán link analysis.
- Phát hiện tài khoản gian lận bằng mô hình học máy.
- Truy vết các pattern rửa tiền phức tạp.
- Trực quan hóa cộng đồng và luồng tiền để hỗ trợ điều tra. :contentReference[oaicite:5]{index=5}

## Cài đặt

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

- tải dữ liệu ở link này: https://drive.google.com/file/d/1ymXYiW3_2mQhirEpyKIRkkns1h5hSSOj/view?usp=sharing
- các setup neo4j server: https://docs.google.com/document/d/1yY-pq6rcXd_glTNKemmdu2i8xFBJHgR1Jw6hauFVL_Q/edit?tab=t.7hys27z4lnq0

## Cấu hình môi trường

Tạo file `.env` hoặc cập nhật cấu hình kết nối Neo4j:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

## Kết quả đầu ra

- Cơ sở dữ liệu đồ thị Neo4j đã chuẩn hóa
- Bảng trust score cho từng account
- Mô hình dự đoán tài khoản gian lận
- Danh sách cộng đồng nghi vấn
- Dashboard trực quan phục vụ review và điều tra

## Định hướng phát triển

- Tối ưu trust score bằng learning-to-rank
- Mở rộng motif detection
- Tích hợp explainability (SHAP / feature importance)
- Xây dựng API phục vụ real-time scoring
- Nâng cấp dashboard cho nghiệp vụ compliance

## Nhóm thực hiện

Dự án được chia thành 5 vai trò chính:

- Data Engineer
- Graph Data Scientist
- Machine Learning Engineer
- AML Graph Miner
- Fullstack / UI-UX Engineer :contentReference[oaicite:6]{index=6}

## Bảng phân chia công việc

| STT | Hạng mục                    | Mô tả công việc                                                                                        |                    Người phụ trách                    |
| :-: | :-------------------------- | :----------------------------------------------------------------------------------------------------- | :---------------------------------------------------: |
|  1  | Data Engineering & ETL      | Làm sạch dữ liệu, chuẩn hóa schema, import dữ liệu vào Neo4j                                           |                          Du                           |
|  2  | Graph Modeling              | Thiết kế node, edge, index, constraint và projection cho GDS                                           |                          Du                           |
|  3  | Trust Scoring               | Xây dựng PageRank, TrustRank, Anti-TrustRank, SimRank, (Nếu có thể thì xây dựng công thức trust_score) |                        Khuyến                         |
|  4  | Fraud Detection ML          | Trích xuất feature, huấn luyện mô hình phân loại fraud/normal                                          | Trung(GNN), Hưng(Boosting thêm với các graph feature) |
|  5  | Graph Mining                | Phát hiện motif, community detection, truy vết cụm nghi vấn                                            |                          Vũ                           |
|  6  | Dashboard & Visualization   | Xây dựng dashboard, biểu đồ, graph visualization, giao diện demo                                       |                          ...                          |
|  7  | Integration & Documentation | Tổng hợp chuẩn bị báo cáo                                                                              |                        Cả nhóm                        |
