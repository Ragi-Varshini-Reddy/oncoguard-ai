"""Build a labeled TCGA-HNSC genomics training table from the GDC API.

This creates the CSV expected by:

    .venv/bin/python -m src.phase3_genomics.train --input data/processed/tcga_hnsc_genomics_training.csv

The label is a prototype stage-derived risk proxy:
- high: AJCC/pathologic stage III or IV, or tumor T3/T4
- low: stage I or II, or tumor T1/T2

Rows without enough staging information are dropped. This is a research/demo
labeling strategy, not a clinically validated risk endpoint.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any
from urllib import parse, request


GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-HNSC"

GENE_PANEL: dict[str, str] = {
    "TP53_expr": "ENSG00000141510",
    "CDKN2A_expr": "ENSG00000147889",
    "EGFR_expr": "ENSG00000146648",
    "PIK3CA_expr": "ENSG00000121879",
    "NOTCH1_expr": "ENSG00000148400",
    "CCND1_expr": "ENSG00000110092",
    "FAT1_expr": "ENSG00000083857",
    "CASP8_expr": "ENSG00000064012",
    "HRAS_expr": "ENSG00000174775",
    "MET_expr": "ENSG00000105976",
    "MYC_expr": "ENSG00000136997",
    "MDM2_expr": "ENSG00000135679",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCGA-HNSC gene-panel training CSV from GDC")
    parser.add_argument("--output", default="data/processed/tcga_hnsc_genomics_training.csv")
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--units", default="median_centered_log2_uqfpkm")
    args = parser.parse_args()

    cases = fetch_cases(max_cases=args.max_cases)
    labeled_cases = [case for case in cases if derive_risk_label(case) is not None]
    print(f"Fetched {len(cases)} TCGA-HNSC cases; {len(labeled_cases)} have usable stage-derived labels")
    if not labeled_cases:
        raise RuntimeError("No labeled cases found; inspect GDC clinical fields or adjust label derivation")

    expression = fetch_expression_values(
        case_ids=[case["case_id"] for case in labeled_cases],
        gene_ids=list(GENE_PANEL.values()),
        units=args.units,
    )
    rows = build_rows(labeled_cases, expression)
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} complete rows created; need at least 20 for current trainer")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "case_id", "risk_label", *GENE_PANEL.keys()]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    high_count = sum(1 for row in rows if row["risk_label"] == "high")
    low_count = sum(1 for row in rows if row["risk_label"] == "low")
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Label counts: high={high_count}, low={low_count}")


def fetch_cases(max_cases: int) -> list[dict[str, Any]]:
    fields = [
        "case_id",
        "submitter_id",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
    ]
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": PROJECT_ID},
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(max_cases),
    }
    payload = get_json(f"{GDC_API}/cases?{parse.urlencode(params)}")
    return payload["data"]["hits"]


def fetch_expression_values(case_ids: list[str], gene_ids: list[str], units: str) -> dict[str, dict[str, float]]:
    payload = {
        "case_ids": case_ids,
        "gene_ids": gene_ids,
        "tsv_units": units,
        "format": "tsv",
    }
    data = post_text(f"{GDC_API}/gene_expression/values", payload, accept="text/tab-separated-values")
    reader = csv.reader(io.StringIO(data), delimiter="\t")
    rows = list(reader)
    if not rows:
        raise RuntimeError("GDC returned no expression rows")
    header = rows[0]
    case_columns = header[1:]
    expression: dict[str, dict[str, float]] = {case_id: {} for case_id in case_columns}
    ensembl_to_feature = {ensembl: feature for feature, ensembl in GENE_PANEL.items()}
    for row in rows[1:]:
        if not row:
            continue
        feature = ensembl_to_feature.get(row[0])
        if feature is None:
            continue
        for case_id, value in zip(case_columns, row[1:], strict=False):
            try:
                expression.setdefault(case_id, {})[feature] = float(value)
            except ValueError:
                expression.setdefault(case_id, {})[feature] = float("nan")
    return expression


def build_rows(cases: list[dict[str, Any]], expression: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        values = expression.get(case_id, {})
        if not all(feature in values for feature in GENE_PANEL):
            continue
        label = derive_risk_label(case)
        if label is None:
            continue
        row: dict[str, Any] = {
            "patient_id": case.get("submitter_id") or case_id,
            "case_id": case_id,
            "risk_label": label,
        }
        for feature in GENE_PANEL:
            row[feature] = values[feature]
        rows.append(row)
    return rows


def derive_risk_label(case: dict[str, Any]) -> str | None:
    diagnoses = case.get("diagnoses") or []
    text_parts: list[str] = []
    for diagnosis in diagnoses:
        for key in ("ajcc_pathologic_stage", "tumor_stage"):
            value = diagnosis.get(key)
            if value:
                text_parts.append(str(value).lower())
    text = " ".join(text_parts)
    if any(token in text for token in ("stage iii", "stage iv", "t3", "t4")):
        return "high"
    if any(token in text for token in ("stage i", "stage ii", "t1", "t2")):
        return "low"
    return None


def get_json(url: str) -> dict[str, Any]:
    with request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def post_text(url: str, payload: dict[str, Any], accept: str) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": accept},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
