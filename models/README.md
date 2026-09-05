# Pre-Trained Model Artifacts Directory

This folder is designated for storing fitted Machine Learning model weights and preprocessor binary files.

## Generated Model Artifacts

When you run `python main.py`, the pipeline automatically trains and saves the following artifacts into this directory:

1. **`models/random_forest_model_tuned.pkl`**: Serialized Tuned Multi-class Random Forest Classifier (200 Estimators, max_depth=25).
2. **`models/kdd_preprocessing_pipeline.pkl`**: Serialized preprocessing dictionary containing `LabelEncoder`, `RobustScaler`, and target mappings.

## Git Exclusion Notice

Large binary pickle files (`*.pkl`) are excluded from Git tracking via `.gitignore` to maintain a lightweight repository footprint (< 100 KB) and avoid GitHub file payload constraints.

### How to Generate Model Artifacts Locally

Execute the training pipeline to build the model binaries locally in seconds:

```bash
# Generate synthetic dataset or use existing data
python dataset_generator.py

# Train model suite and save artifacts to models/
python main.py --data data/synthetic_test_dataset.csv
```
