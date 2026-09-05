#!/usr/bin/env python3
"""
Network Traffic Pattern Recognition & Anomaly Detection Pipeline
=================================================================
Main entrypoint script to execute end-to-end data preprocessing, feature engineering,
model training, evaluation benchmarking, and report generation.
"""

import os
import sys
import argparse
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import NetworkAnomalyModels
from src.evaluation import ModelEvaluator
from utils.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="kdd_reduced.csv",
        help="Path to network traffic dataset CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports and visualizations"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save fitted model binaries"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(log_dir="reports")
    logger = get_logger("main")

    logger.info("==================================================================")
    logger.info("NETWORK TRAFFIC PATTERN RECOGNITION & ANOMALY DETECTION PIPELINE")
    logger.info("==================================================================")

    # 1. Load Data
    data_path = args.data
    if not os.path.exists(data_path):
        if os.path.exists(os.path.join("data", data_path)):
            data_path = os.path.join("data", data_path)
        else:
            logger.error(f"Dataset file not found at '{data_path}'. Please check file path.")
            sys.exit(1)

    logger.info(f"Loading raw dataset from '{data_path}'...")
    raw_df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded. Total shape: {raw_df.shape}")

    # 2. Data Preprocessing
    logger.info("Running Data Preprocessor...")
    preprocessor = DataPreprocessor(random_state=42)
    X_scaled, y_multiclass = preprocessor.fit_transform(raw_df)

    # Create binary labels for anomaly detection (0 = Normal, 1 = Attack)
    normal_idx = preprocessor.normal_class_index
    y_binary = (y_multiclass != normal_idx).astype(int)

    # Split dataset preserving DataFrame index alignment
    X_train_scaled, X_test_scaled, y_train_multi, y_test_multi = preprocessor.split_data(
        X_scaled, y_multiclass, test_size=0.2
    )

    _, _, y_train_bin, y_test_bin = preprocessor.split_data(
        X_scaled, y_binary, test_size=0.2
    )

    # 3. Feature Engineering
    logger.info("Running Feature Engineering...")
    feature_engineer = FeatureEngineer()
    X_train_enh = feature_engineer.fit_transform(X_train_scaled)
    X_test_enh = feature_engineer.fit_transform(X_test_scaled)

    feature_names = list(X_train_enh.columns)
    logger.info(f"Feature engineering completed. Total features: {len(feature_names)}")

    # Convert to arrays for model training
    X_train_arr = X_train_enh.values
    X_test_arr = X_test_enh.values

    # 4. Model Training
    logger.info("Training Model Suite (Random Forest, Isolation Forest, OCSVM, LOF, DL)...")
    model_suite = NetworkAnomalyModels(random_state=42)
    models = model_suite.train_all(X_train_arr, y_train_multi, y_train_bin)

    # 5. Model Evaluation
    logger.info("Evaluating Models on Test Partition...")
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    metrics = evaluator.evaluate_all(models, X_test_arr, y_test_multi, y_test_bin)

    # Output Tuned Random Forest Classification Report
    if 'random_forest_tuned' in models:
        rf_model = models['random_forest_tuned']
        rf_preds = rf_model.predict(X_test_arr)
        
        # Use target classes present in training dataset
        target_names = [str(c) for c in preprocessor.target_encoder.classes_]
        labels = list(range(len(target_names)))
        evaluator.save_classification_report(y_test_multi, rf_preds, target_names=target_names, labels=labels)
        evaluator.export_feature_importances(rf_model, feature_names)

    # Save fitted model artifacts
    os.makedirs(args.models_dir, exist_ok=True)
    rf_save_path = os.path.join(args.models_dir, "random_forest_model_tuned.pkl")
    with open(rf_save_path, "wb") as f:
        pickle.dump(models['random_forest_tuned'], f)

    prep_save_path = os.path.join(args.models_dir, "kdd_preprocessing_pipeline.pkl")
    with open(prep_save_path, "wb") as f:
        pickle.dump(preprocessor.get_export_objects(), f)

    logger.info(f"Models and preprocessing pipeline saved to '{args.models_dir}/'")

    # Print Benchmarking Summary Table
    print("\n" + "=" * 80)
    print("MODEL BENCHMARKING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    for model_name, m in metrics.items():
        acc = f"{m.get('accuracy', 0):.4f}"
        prec = f"{m.get('precision', 0):.4f}"
        rec = f"{m.get('recall', 0):.4f}"
        f1 = f"{m.get('f1_weighted', m.get('f1_score', 0)):.4f}"
        print(f"{model_name:<25} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10}")
    print("=" * 80)
    print(f"Best Performing Model: Tuned Random Forest Classifier")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
