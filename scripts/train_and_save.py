"""Train models and save artifacts to models/."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import save_artifacts, train_pipeline


def main():
    print("Training pipeline...")
    artifacts = train_pipeline(str(PROJECT_ROOT))
    out_dir = save_artifacts(artifacts, str(PROJECT_ROOT))
    metrics = artifacts["metrics"]
    print(f"Saved artifacts to {out_dir}")
    print(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"XGBoost churn recall: {metrics['churn_recall']:.4f}")
    print(f"Baseline ROC-AUC: {metrics['baseline']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
