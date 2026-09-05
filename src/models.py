#!/usr/bin/env python3
"""
Model Architecture Module for Network Traffic Anomaly Detection
===============================================================
Defines supervised ensemble models, unsupervised anomaly detectors, and deep learning architectures.
"""

from typing import Dict, Any, Optional
import numpy as np

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class NetworkAnomalyModels:
    """Manages creation, training, and prediction for all ML/DL models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def build_tuned_random_forest(self) -> RandomForestClassifier:
        """Build tuned multi-class Random Forest Classifier."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )

    def build_isolation_forest(self, contamination: float = 0.1) -> IsolationForest:
        """Build Isolation Forest anomaly detector."""
        return IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def build_one_class_svm(self, nu: float = 0.1) -> OneClassSVM:
        """Build One-Class SVM anomaly detector."""
        return OneClassSVM(kernel='rbf', gamma='scale', nu=nu)

    def build_lof(self, contamination: float = 0.1) -> LocalOutlierFactor:
        """Build Local Outlier Factor anomaly detector (novelty=True for test prediction)."""
        return LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)

    def build_deep_learning_model(self, input_dim: int) -> Optional[Any]:
        """Build 3-layer Dense Neural Network using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        inp = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model

    def train_all(
        self, X_train: np.ndarray, y_train_multiclass: np.ndarray, y_train_binary: np.ndarray
    ) -> Dict[str, Any]:
        """Train all models on the dataset."""
        # 1. Tuned Random Forest (Multi-class Classification)
        rf_tuned = self.build_tuned_random_forest()
        rf_tuned.fit(X_train, y_train_multiclass)
        self.models['random_forest_tuned'] = rf_tuned

        # 2. Isolation Forest (Binary Anomaly Detection)
        iso_forest = self.build_isolation_forest()
        iso_forest.fit(X_train)
        self.models['isolation_forest'] = iso_forest

        # 3. One-Class SVM (Binary Anomaly Detection)
        oc_svm = self.build_one_class_svm()
        oc_svm.fit(X_train)
        self.models['one_class_svm'] = oc_svm

        # 4. Local Outlier Factor (Novelty Detection)
        lof = self.build_lof()
        lof.fit(X_train)
        self.models['lof'] = lof

        # 5. Deep Learning (Binary Anomaly Classification)
        if TENSORFLOW_AVAILABLE:
            dl_model = self.build_deep_learning_model(X_train.shape[1])
            if dl_model is not None:
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5)
                ]
                dl_model.fit(
                    X_train,
                    y_train_binary,
                    epochs=50,
                    batch_size=1024,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                self.models['deep_learning'] = dl_model

        return self.models
