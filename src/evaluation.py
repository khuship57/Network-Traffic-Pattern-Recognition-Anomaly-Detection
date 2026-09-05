#!/usr/bin/env python3
"""
Evaluation & Visualization Module for Network Traffic Pattern Recognition
==========================================================================
Computes quantitative evaluation metrics and renders plots for model benchmarking.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)


class ModelEvaluator:
    """Evaluates trained models and generates report artifacts."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics: Dict[str, Any] = {}

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: ModelEvaluator._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ModelEvaluator._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test_multiclass: np.ndarray,
        y_test_binary: np.ndarray
    ) -> Dict[str, Any]:
        """Compute performance metrics for all fitted models."""
        for name, model in models.items():
            try:
                if name == 'random_forest_tuned':
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test_multiclass, y_pred)
                    prec = precision_score(y_test_multiclass, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test_multiclass, y_pred, average='weighted', zero_division=0)
                    f1_w = f1_score(y_test_multiclass, y_pred, average='weighted', zero_division=0)
                    f1_m = f1_score(y_test_multiclass, y_pred, average='macro', zero_division=0)

                    self.metrics[name] = {
                        'accuracy': float(acc),
                        'precision': float(prec),
                        'recall': float(rec),
                        'f1_weighted': float(f1_w),
                        'f1_macro': float(f1_m),
                        'predictions': y_pred.tolist()
                    }

                elif name == 'deep_learning':
                    y_prob = model.predict(X_test, verbose=0).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                    acc = accuracy_score(y_test_binary, y_pred)
                    prec = precision_score(y_test_binary, y_pred, zero_division=0)
                    rec = recall_score(y_test_binary, y_pred, zero_division=0)
                    f1 = f1_score(y_test_binary, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test_binary, y_prob)

                    self.metrics[name] = {
                        'accuracy': float(acc),
                        'precision': float(prec),
                        'recall': float(rec),
                        'f1_score': float(f1),
                        'auc': float(auc)
                    }

                elif name in ['isolation_forest', 'one_class_svm', 'lof']:
                    # Anomaly detectors return -1 for anomaly (attack), 1 for normal
                    y_pred_raw = model.predict(X_test)
                    y_pred = (y_pred_raw == -1).astype(int)
                    
                    acc = accuracy_score(y_test_binary, y_pred)
                    prec = precision_score(y_test_binary, y_pred, zero_division=0)
                    rec = recall_score(y_test_binary, y_pred, zero_division=0)
                    f1 = f1_score(y_test_binary, y_pred, zero_division=0)

                    self.metrics[name] = {
                        'accuracy': float(acc),
                        'precision': float(prec),
                        'recall': float(rec),
                        'f1_score': float(f1)
                    }

            except Exception as e:
                print(f"Error evaluating {name}: {e}")

        return self.metrics

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Generate and save JSON classification report with explicit labels."""
        if labels is None and target_names is not None:
            labels = list(range(len(target_names)))

        report_dict = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
        )
        report_path = os.path.join(self.output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(self._to_serializable(report_dict), f, indent=2)
        return report_dict

    def export_feature_importances(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract and save feature importances to CSV."""
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)

            out_path = os.path.join(self.output_dir, 'feature_importances.csv')
            fi_df.to_csv(out_path, index=False)
            return fi_df
        return pd.DataFrame()
