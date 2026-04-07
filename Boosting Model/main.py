from src.load_data import load_data
from src.preprocess import preprocess, feature_engineering, prepare_xy
from src.graph_features import build_graph_features
from src.train import train_model
from src.evaluate import evaluate
from src.predict import predict

def main():
    print("Load data...")
    features, nodes, communities, fraud_accounts, cycles = load_data()
    print("Preprocess...")
    df = preprocess(features, nodes)
    print("Graph features...")
    df = build_graph_features(df, communities, fraud_accounts, cycles)
    print("Feature engineering...")
    df = feature_engineering(df)
    print("Prepare data...")
    X, y = prepare_xy(df)
    print("Train model...")
    lgb_model, rf_model, X_test, y_test = train_model(X, y)
    print("Evaluate...")
    evaluate(lgb_model, rf_model, X_test, y_test)
    print("Predict full dataset...")
    result = predict(df, X)
    result.to_csv("final_output.csv", index=False)
    print("Saved: final_output.csv")
    print("DONE!")
if __name__ == "__main__":
    main()