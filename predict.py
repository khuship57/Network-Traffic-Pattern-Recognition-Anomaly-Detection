#!/usr/bin/env python3
"""
Model Inference Script for Network Traffic Pattern Recognition
==============================================================
Loads pre-trained Random Forest Classifier and preprocessing pipeline to predict
intrusion classes on new/unseen network connection logs.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict intrusion attack classes on network traffic logs.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/Test.csv",
        help="Path to input network traffic CSV file for prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/random_forest_model_tuned.pkl",
        help="Path to trained model pickle file"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="models/kdd_preprocessing_pipeline.pkl",
        help="Path to preprocessing objects pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/predictions.csv",
        help="Path to save output predictions CSV file"
    )
    return parser.parse_args()


def load_pickle(path: str):
    """Load pickle file with fallback search paths."""
    candidates = [
        path,
        os.path.join("models", os.path.basename(path)),
        os.path.join("Model", os.path.basename(path)),
        os.path.basename(path)
    ]
    
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                last_err = e
                continue

    raise FileNotFoundError(f"Could not load pickle file from '{path}'. Error: {last_err}")


def main():
    args = parse_args()

    print("=" * 70)
    print("NETWORK INTRUSION PREDICTION ENGINE")
    print("=" * 70)

    # Load model and preprocessor objects
    print(f"Loading model from '{args.model}'...")
    try:
        model = load_pickle(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Loading preprocessing objects from '{args.preprocessor}'...")
    try:
        prep = load_pickle(args.preprocessor)
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        sys.exit(1)

    # Load input data
    if not os.path.exists(args.input):
        print(f"Input file not found at '{args.input}'")
        sys.exit(1)

    input_df = pd.read_csv(args.input)
    print(f"Loaded input logs shape: {input_df.shape}")

    # Remove target/meta columns if present
    X = input_df.drop(columns=['class', 'difficulty', 'index', 'Unnamed: 0'], errors='ignore')

    # Handle dictionary vs class instance for preprocessor
    if isinstance(prep, dict):
        label_encoders = prep.get('label_encoders', {})
        scaler = prep.get('scaler', None)
        target_encoder = prep.get('target_encoder', None)
    else:
        label_encoders = getattr(prep, 'label_encoders', {})
        scaler = getattr(prep, 'scaler', None)
        target_encoder = getattr(prep, 'target_encoder', None)

    categorical_cols = ['protocol_type', 'service', 'flag']

    # Apply categorical encoding
    for col in categorical_cols:
        if col in X.columns and col in label_encoders:
            le = label_encoders[col]
            X[col] = X[col].astype(str).map(
                lambda val: le.transform([val])[0] if hasattr(le, 'classes_') and val in le.classes_ else -1
            )

    # Scale numerical features
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            X[numerical_cols] = scaler.transform(X[numerical_cols])
        except Exception:
            pass

    # Feature Engineering (Interaction features: total_bytes, byte_ratio, connection_intensity, service_diversity)
    feature_engineer = FeatureEngineer()
    X = feature_engineer.fit_transform(X)

    # Run Prediction
    print("Running predictions...")
    predictions = model.predict(X.values)
    probabilities = model.predict_proba(X.values)

    # Map back target class names if available
    if target_encoder is not None and hasattr(target_encoder, 'inverse_transform'):
        try:
            predicted_classes = target_encoder.inverse_transform(predictions)
        except Exception:
            predicted_classes = predictions
    else:
        predicted_classes = predictions

    # Build output DataFrame
    output_df = input_df.copy()
    output_df['predicted_class'] = predicted_classes
    output_df['confidence'] = np.max(probabilities, axis=1)

    # Save outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: '{args.output}'")
    print("=" * 70)
    print("\nPrediction Summary:")
    print(output_df[['predicted_class', 'confidence']].head(10))


if __name__ == "__main__":
    main()
