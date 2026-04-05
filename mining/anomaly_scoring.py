import pandas as pd
import os

def calculate_risk_scores():
    print(" Đang bắt đầu tổng hợp điểm rủi ro...")

    # 1. Load các kết quả đã tìm được từ các bước trước
    try:
        df_cycles = pd.read_csv("suspicious_cycles.csv")
        df_classified = pd.read_csv("classified_fraud_accounts.csv")
        df_communities = pd.read_csv("account_communities.csv")
    except FileNotFoundError as e:
        print(f" Thiếu file kết quả!  {e}")
        return

    # 2. Xử lý file Vòng lặp (Cycles)
    # Vì một tài khoản có thể nằm ở cột A, B hoặc C, ta gom hết lại để đếm số lần xuất hiện
    all_cycle_ids = pd.concat([df_cycles['Account_A'], df_cycles['Account_B'], df_cycles['Account_C']])
    cycle_counts = all_cycle_ids.value_counts().reset_index()
    cycle_counts.columns = ['AccountID', 'Cycle_Occurrence']

    # 3. Chuẩn bị bảng điểm tổng hợp từ danh sách cộng đồng (tất cả tài khoản)
    risk_table = df_communities.copy()
    risk_table.rename(columns={'AccountID': 'AccountID'}, inplace=True)
    
    # 4. Tích hợp điểm từ Vòng lặp (Mỗi lần tham gia vòng lặp +5 điểm)
    risk_table = risk_table.merge(cycle_counts, on='AccountID', how='left').fillna(0)
    risk_table['Score_Cycle'] = risk_table['Cycle_Occurrence'] * 5

    # 5. Tích hợp điểm từ Phân loại Fan-in/Fan-out (+10 điểm nếu bị gắn nhãn)
    classified_ids = df_classified[['Suspicious_Account', 'Fraud_Type']].copy()
    classified_ids.columns = ['AccountID', 'Fraud_Type']
    risk_table = risk_table.merge(classified_ids, on='AccountID', how='left')
    risk_table['Score_Fraud_Type'] = risk_table['Fraud_Type'].apply(lambda x: 10 if pd.notnull(x) else 0)

    # 6. Tính tổng điểm Risk Score
    risk_table['Final_Risk_Score'] = risk_table['Score_Cycle'] + risk_table['Score_Fraud_Type']

    # 7. Lọc (Score > 0)
    high_risk_accounts = risk_table[risk_table['Final_Risk_Score'] > 0].sort_values(by='Final_Risk_Score', ascending=False)

    print("\n TOP 10 TÀI KHOẢN CÓ ĐIỂM RỦI RO CAO NHẤT")
    print(high_risk_accounts[['AccountID', 'communityId', 'Final_Risk_Score', 'Fraud_Type']].head(10))

    # 8. Xuất file cuối cùng 
    high_risk_accounts.to_csv("final_risk_scoring.csv", index=False)
    print("\n Đã tạo file 'final_risk_scoring.csv'.")

if __name__ == "__main__":
    calculate_risk_scores()