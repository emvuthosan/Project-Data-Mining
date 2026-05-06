import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from src.load_data import load_data
from src.preprocess import preprocess, feature_engineering, prepare_xy
from src.graph_features import build_graph_features
from src.rels_features import build_rels_features
from src.train import train_model
from src.evaluate import evaluate
from src.predict import predict

def main():
    # Set working directory to Boosting Model folder so outputs go to correct place
    base_dir = os.path.dirname(__file__)
    if base_dir:
        os.chdir(base_dir)

    print("Load data...")
    df = load_data()
    print("Preprocess...")
    df = preprocess(df)
    print("Graph features...")
    df = build_graph_features(df)
    print("Transaction features (rels)...")
    rels_feats = build_rels_features()
    if rels_feats is not None:
        df = df.merge(rels_feats, on='node_id', how='left')
        df = df.fillna(0)
        print(f"  Merged {len(rels_feats.columns) - 1} rels features into main DataFrame.")
    print("Feature engineering...")
    df = feature_engineering(df)
    print("Prepare data...")
    X, y = prepare_xy(df)
    # Giải phóng bộ nhớ trước khi train
    import gc
    del rels_feats
    gc.collect()
    print(f"  Features: {X.shape[1]} cột, {X.shape[0]} dòng, Memory: {X.memory_usage(deep=True).sum() / 1024**2:.0f} MB")
    print("Train model...")
    lgb_models, rf_models, optimal_threshold, X_test, y_test = train_model(X, y)
    print("Evaluate...")
    evaluate(lgb_models, rf_models, optimal_threshold, X_test, y_test)
    print("Predict full dataset...")
    result = predict(df, X)
    result.to_csv("final_output.csv", index=False)
    print("Saved: final_output.csv")
    print("DONE!")

if __name__ == "__main__":
    main()