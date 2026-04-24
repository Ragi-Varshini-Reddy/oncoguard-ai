"""Artifact-backed genomics model for real-data training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.utils import clamp, risk_class_from_score


@dataclass
class GenomicsArtifacts:
    feature_order: list[str]
    pipeline: Pipeline
    positive_label: str
    negative_labels: list[str]
    embedding_model: PCA | None
    embedding_dim: int
    metrics: dict[str, float | str | int | None]
    model_card: dict[str, Any]


def train_genomics_artifacts(
    training_table_path: str | Path,
    config: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Train a real artifact-backed genomics classifier from labeled rows."""

    table = _read_table(training_table_path)
    training_cfg = config["genomics"].get("training", {})
    label_column = training_cfg.get("label_column", "risk_label")
    patient_id_column = training_cfg.get("patient_id_column", "patient_id")
    positive_label = str(training_cfg.get("positive_label", "high"))
    feature_order = list(config["genomics"]["selected_gene_panel"])
    missing_columns = [column for column in [patient_id_column, label_column, *feature_order] if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Training table missing required columns: {missing_columns}")

    min_rows = int(training_cfg.get("min_training_rows", 20))
    if len(table) < min_rows:
        raise ValueError(f"Need at least {min_rows} labeled rows for real training; got {len(table)}")

    x = table[feature_order].apply(pd.to_numeric, errors="coerce")
    y_raw = table[label_column].astype(str)
    if positive_label not in set(y_raw):
        raise ValueError(f"Positive label {positive_label!r} not found in {label_column}")
    y = (y_raw == positive_label).astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        raise ValueError("Training labels must contain at least two classes")

    test_size = float(training_cfg.get("test_size", 0.2))
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=int(config["project"].get("random_seed", 468)),
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=int(config["project"].get("random_seed", 468)),
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    transformed_train = pipeline[:-1].transform(x_train)
    pca_components = min(int(config["genomics"].get("embedding_dim", 128)), transformed_train.shape[0], transformed_train.shape[1])
    embedding_model = PCA(n_components=pca_components, random_state=int(config["project"].get("random_seed", 468)))
    embedding_model.fit(transformed_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | str | int | None] = {
        "rows": int(len(table)),
        "features": len(feature_order),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": _safe_auc(y_test, probabilities),
        "label_column": label_column,
        "positive_label": positive_label,
    }
    artifacts = GenomicsArtifacts(
        feature_order=feature_order,
        pipeline=pipeline,
        positive_label=positive_label,
        negative_labels=sorted(label for label in set(y_raw) if label != positive_label),
        embedding_model=embedding_model,
        embedding_dim=int(config["genomics"].get("embedding_dim", 128)),
        metrics=metrics,
        model_card={
            "model_type": "logistic_regression_gene_panel",
            "preprocessing": ["median_imputation_fit_on_training", "standard_scaling_fit_on_training"],
            "xai": "linear_coefficient_x_standardized_value",
            "intended_use": "research prototype risk-support model; requires validation before clinical use",
            "data_source": str(training_table_path),
        },
    )
    output_path = Path(artifact_path or config["genomics"].get("artifact_path", "artifacts/genomics_model.joblib"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)
    return output_path, {"metrics": metrics, "model_card": artifacts.model_card}


def load_genomics_artifacts(config: dict[str, Any], artifact_path: str | Path | None = None) -> GenomicsArtifacts:
    path = Path(artifact_path or config["genomics"].get("artifact_path", "artifacts/genomics_model.joblib"))
    if not path.exists():
        raise FileNotFoundError(f"Genomics model artifact not found at {path}")
    return joblib.load(path)


def infer_with_genomics_artifacts(
    patient_id: str,
    genomic_features: dict[str, Any],
    config: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_genomics_artifacts(config, artifact_path)
    frame = pd.DataFrame([{feature: genomic_features.get(feature) for feature in artifacts.feature_order}])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    missing_features = [feature for feature in artifacts.feature_order if pd.isna(frame.loc[0, feature])]
    transformed = artifacts.pipeline[:-1].transform(frame)
    probability = float(artifacts.pipeline.predict_proba(frame)[0, 1])
    risk_score = clamp(probability)
    raw_embedding = artifacts.embedding_model.transform(transformed)[0].tolist() if artifacts.embedding_model else []
    embedding = _pad_embedding(raw_embedding, artifacts.embedding_dim)
    top_features = explain_artifact_prediction(artifacts, transformed[0], frame.iloc[0].to_dict())
    confidence = clamp(0.55 + abs(risk_score - 0.5) * 0.75 - (len(missing_features) / max(1, len(artifacts.feature_order))) * 0.25)
    return {
        "patient_id": patient_id,
        "embedding": embedding,
        "embedding_dim": artifacts.embedding_dim,
        "risk_score": round(risk_score, 4),
        "risk_class": risk_class_from_score(risk_score),
        "diagnosis_probability": round(risk_score, 4),
        "confidence": round(confidence, 4),
        "missing_features": missing_features,
        "top_features": top_features,
        "metrics": artifacts.metrics,
        "model_card": artifacts.model_card,
    }


def explain_artifact_prediction(
    artifacts: GenomicsArtifacts,
    standardized_values: np.ndarray,
    raw_values: dict[str, Any],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    classifier = artifacts.pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    rows: list[dict[str, Any]] = []
    for feature, raw_value, standardized_value, coefficient in zip(
        artifacts.feature_order,
        raw_values.values(),
        standardized_values,
        coefficients,
        strict=True,
    ):
        effect = float(standardized_value * coefficient)
        direction = "neutral"
        if effect > 0.0:
            direction = "increases_risk"
        elif effect < 0.0:
            direction = "decreases_risk"
        rows.append(
            {
                "feature": feature,
                "value": None if pd.isna(raw_value) else float(raw_value),
                "standardized_value": round(float(standardized_value), 4),
                "coefficient": round(float(coefficient), 4),
                "importance_score": round(abs(effect), 4),
                "direction": direction,
            }
        )
    rows.sort(key=lambda item: item["importance_score"], reverse=True)
    return rows[:top_k]


def _read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        return round(float(roc_auc_score(y_true, probabilities)), 4)
    except ValueError:
        return None


def _pad_embedding(values: list[float], dim: int) -> list[float]:
    padded = [round(float(value), 6) for value in values[:dim]]
    padded.extend([0.0] * max(0, dim - len(padded)))
    return padded
