import pandas as pd
import mlflow

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Research Paper Recommender")

# Load evaluation results
results = pd.read_csv("data/evaluation_results.csv")


# =========================
# TF-IDF Experiment
# =========================

with mlflow.start_run(run_name="tfidf_evaluation"):

    mlflow.log_param("model", "TF-IDF")
    mlflow.log_param("max_features", 50000)
    mlflow.log_param("original_dataset_size", 41127)
    mlflow.log_param("evaluation_dataset_size", 41120)
    mlflow.log_param("num_queries", 100)
    mlflow.log_param(
        "relevance_definition",
        "shared_arxiv_terms"
    )

    tfidf_metrics = {
        "precision_at_10": results.loc[
            results["Metric"] == "Precision@10",
            "TF-IDF"
        ].iloc[0],

        "recall_at_10": results.loc[
            results["Metric"] == "Recall@10",
            "TF-IDF"
        ].iloc[0],

        "ndcg_at_10": results.loc[
            results["Metric"] == "NDCG@10",
            "TF-IDF"
        ].iloc[0],

        "mrr_at_10": results.loc[
            results["Metric"] == "MRR@10",
            "TF-IDF"
        ].iloc[0],

        "map_at_10": results.loc[
            results["Metric"] == "AP@10",
            "TF-IDF"
        ].iloc[0],
    }

    mlflow.log_metrics({
        key: float(value)
        for key, value in tfidf_metrics.items()
    })


# =========================
# Transformer Experiment
# =========================

with mlflow.start_run(run_name="transformer_evaluation"):

    mlflow.log_param(
        "model",
        "all-MiniLM-L6-v2"
    )

    mlflow.log_param(
        "embedding_dimension",
        384
    )

    mlflow.log_param(
        "original_dataset_size",
        41127
    )

    mlflow.log_param(
        "evaluation_dataset_size",
        41120
    )

    mlflow.log_param(
        "num_queries",
        100
    )

    mlflow.log_param(
        "relevance_definition",
        "shared_arxiv_terms"
    )

    transformer_metrics = {
        "precision_at_10": results.loc[
            results["Metric"] == "Precision@10",
            "Transformer"
        ].iloc[0],

        "recall_at_10": results.loc[
            results["Metric"] == "Recall@10",
            "Transformer"
        ].iloc[0],

        "ndcg_at_10": results.loc[
            results["Metric"] == "NDCG@10",
            "Transformer"
        ].iloc[0],

        "mrr_at_10": results.loc[
            results["Metric"] == "MRR@10",
            "Transformer"
        ].iloc[0],

        "map_at_10": results.loc[
            results["Metric"] == "AP@10",
            "Transformer"
        ].iloc[0],
    }

    mlflow.log_metrics({
        key: float(value)
        for key, value in transformer_metrics.items()
    })


print("MLflow experiments completed successfully.")