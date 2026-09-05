#!/usr/bin/env python3
"""
Data Preprocessing Module for Network Traffic Pattern Recognition
================================================================
Handles data cleaning, categorical label encoding, RobustScaler numeric normalization,
and stratified train-test splitting.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


class DataPreprocessor:
    """Preprocesses raw network traffic dataset for training and inference."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: RobustScaler = RobustScaler()
        self.target_encoder: LabelEncoder = LabelEncoder()
        self.categorical_features = ['protocol_type', 'service', 'flag']
        self.numerical_features = []
        self.normal_class_index = 0

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit preprocessor objects and transform raw dataset."""
        data = df.copy()

        # Drop index or unnamed columns if present
        if data.columns[0] in ['Unnamed: 0', 'index', '']:
            data = data.drop(data.columns[0], axis=1)

        y = data['class'].copy()
        X = data.drop(columns=['class', 'difficulty'], errors='ignore')

        # Fill missing numerical values with median
        X = X.fillna(X.median(numeric_only=True))

        self.numerical_features = [c for c in X.columns if c not in self.categorical_features]

        # Encode categorical features
        X_encoded = X.copy()
        for col in self.categorical_features:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le

        # Scale numerical features
        X_scaled = X_encoded.copy()
        if self.numerical_features:
            X_scaled[self.numerical_features] = self.scaler.fit_transform(X_encoded[self.numerical_features])

        # Encode target variable
        y_encoded = self.target_encoder.fit_transform(y)
        if 'normal' in list(self.target_encoder.classes_):
            self.normal_class_index = int(np.where(self.target_encoder.classes_ == 'normal')[0][0])

        return X_scaled, y_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test/inference dataset using fitted preprocessors."""
        data = df.copy()

        if data.columns[0] in ['Unnamed: 0', 'index', '']:
            data = data.drop(data.columns[0], axis=1)

        X = data.drop(columns=['class', 'difficulty'], errors='ignore')
        X = X.fillna(X.median(numeric_only=True))

        X_encoded = X.copy()
        for col in self.categorical_features:
            if col in X_encoded.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categorical labels gracefully
                X_encoded[col] = X_encoded[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )

        X_scaled = X_encoded.copy()
        if self.numerical_features and hasattr(self.scaler, 'center_'):
            X_scaled[self.numerical_features] = self.scaler.transform(X_encoded[self.numerical_features])

        return X_scaled

    def split_data(
        self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train and test sets while preserving DataFrame indices."""
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

    def get_export_objects(self) -> Dict[str, Any]:
        """Return fitted preprocessing objects for serialization."""
        return {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'normal_class_index': self.normal_class_index,
        }
