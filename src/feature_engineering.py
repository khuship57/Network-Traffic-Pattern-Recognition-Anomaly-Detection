#!/usr/bin/env python3
"""
Feature Engineering Module for Network Traffic Pattern Recognition
===================================================================
Constructs domain-specific network traffic features while maintaining exact row alignment.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Extracts additional interaction features from network traffic attributes."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers interaction features on the given dataset."""
        X_enh = df.copy()

        # Aggregate byte interaction metrics
        if 'src_bytes' in X_enh.columns and 'dst_bytes' in X_enh.columns:
            X_enh['total_bytes'] = X_enh['src_bytes'] + X_enh['dst_bytes']
            X_enh['byte_ratio'] = X_enh['src_bytes'] / (X_enh['dst_bytes'] + 1.0)

        # Connection intensity and service diversity
        if 'count' in X_enh.columns and 'srv_count' in X_enh.columns:
            X_enh['connection_intensity'] = X_enh['count'] + X_enh['srv_count']
            X_enh['service_diversity'] = X_enh['srv_count'] / (X_enh['count'] + 1.0)

        # Replace any inf, -inf, or nan generated during feature calculations
        X_enh = X_enh.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return X_enh
