"""Patient-facing query responses grounded in available model outputs."""

from __future__ import annotations

from typing import Any

from backend.ml.common.contracts import FusionOutput, ModuleOutput
from backend.ml.explainability.llm_provider import generate_patient_answer


DISCLAIMER = (
    "This is decision-support information from the prototype, not a diagnosis. "
    "Please review it with a qualified clinician."
)


def answer_patient_query(
    patient_id: str,
    query: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    config: dict[str, Any] | None = None,
    use_llm: bool = True,
    history: list[dict[str, str]] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _is_greeting(query):
        return _chatty_response(
            patient_id,
            query,
            "Hi, I am OralCare-AI. You can ask me about your risk status, what influenced the result, missing information, or what to discuss with your doctor.",
            module_outputs,
            fusion_output,
            "greeting",
        )

    context = _patient_context(patient_id, module_outputs, fusion_output, extra_context)
    if config:
        context["llm_system_prompt"] = config.get("llm", {}).get("system_prompt")
    llm_cfg = (config or {}).get("llm", {})
    if use_llm and llm_cfg.get("enabled", False):
        try:
            llm_result = generate_patient_answer(query, context, config or {}, history=history)
            return {
                "patient_id": patient_id,
                "query": query,
                "answer": llm_result.text,
                "disclaimer": DISCLAIMER,
                "sources_used": _sources_used(module_outputs, fusion_output),
                "suggested_questions": _suggested_questions(),
                "answer_mode": "llm",
                "llm_provider": llm_result.provider,
                "llm_model": llm_result.model,
            }
        except Exception as exc:
            if not llm_cfg.get("allow_rule_based_fallback", True):
                raise
            fallback_warning = f"LLM unavailable, used deterministic fallback: {exc}"
        else:
            fallback_warning = ""
    else:
        fallback_warning = "LLM disabled; used deterministic fallback"

    normalized = query.lower()
    sections: list[str] = []

    if _mentions(normalized, ("why", "explain", "reason", "because", "feature", "gene")):
        sections.append(_xai_summary(module_outputs, fusion_output))
    if _mentions(normalized, ("risk", "status", "health", "score", "condition", "how am i")):
        sections.append(_risk_summary(fusion_output, module_outputs))
    if _mentions(normalized, ("suggest", "advice", "next", "should", "recommend", "do now", "steps")):
        sections.append(_suggestions(module_outputs, fusion_output))
    if _mentions(normalized, ("missing", "available", "provided", "modality", "data")):
        sections.append(_availability(module_outputs, fusion_output))

    if len(sections) == 0:
        sections.extend([_risk_summary(fusion_output, module_outputs), _suggestions(module_outputs, fusion_output)])

    answer = "\n\n".join(section for section in sections if section)
    return {
        "patient_id": patient_id,
        "query": query,
        "answer": answer,
        "disclaimer": DISCLAIMER,
        "sources_used": _sources_used(module_outputs, fusion_output),
        "suggested_questions": _suggested_questions(),
        "answer_mode": "rule_based",
        "llm_provider": None,
        "llm_model": None,
        "fallback_warning": fallback_warning,
    }


def answer_patient_chat(
    patient_id: str,
    message: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    history: list[dict[str, str]],
    config: dict[str, Any] | None = None,
    use_llm: bool = True,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = answer_patient_query(
        patient_id,
        message,
        module_outputs,
        fusion_output,
        config=config,
        use_llm=use_llm,
        history=history,
        extra_context=extra_context,
    )
    return {
        **answer,
        "message": message,
    }


def _summary(
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
) -> str:
    if fusion_output:
        diagnosis = fusion_output.diagnosis.model_dump(by_alias=True)
        risk = fusion_output.risk.model_dump(by_alias=True)
        return (
            f"For patient {patient_id}, the current fused prototype output is "
            f"{risk['class']} risk ({risk['score']:.1%}) with {fusion_output.confidence:.1%} confidence. "
            f"The diagnosis-support class is {diagnosis['class']} ({diagnosis['probability']:.1%})."
        )
    available = [output.modality for output in module_outputs if output.status == "available"]
    if available:
        return f"For patient {patient_id}, available prototype information currently includes: {', '.join(available)}."
    return f"For patient {patient_id}, no available modality output was provided yet."


def _risk_summary(fusion_output: FusionOutput | None, module_outputs: list[ModuleOutput]) -> str:
    if fusion_output:
        risk = fusion_output.risk.model_dump(by_alias=True)
        return (
            f"Health status summary: the fused risk output is {risk['class']} ({risk['score']:.1%}). "
            "This should be treated as a prioritization signal for clinician review, not a final medical conclusion."
        )

    available_predictions = [
        (output.modality, output.prediction or {}, output.confidence)
        for output in module_outputs
        if output.status == "available"
    ]
    if not available_predictions:
        return "Health status summary: there is not enough available model output to summarize risk yet."
    parts = []
    for modality, prediction, confidence in available_predictions:
        risk_score = prediction.get("risk_score", prediction.get("diagnosis_probability"))
        if isinstance(risk_score, (int, float)):
            parts.append(f"{modality}: {risk_score:.1%} risk signal, confidence {confidence or 0:.1%}")
    return "Standalone modality signals: " + "; ".join(parts) + "."


def _xai_summary(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> str:
    lines: list[str] = []
    if fusion_output and fusion_output.modality_contributions:
        ranked = sorted(fusion_output.modality_contributions.items(), key=lambda item: item[1], reverse=True)
        active = [f"{modality} ({score:.1%})" for modality, score in ranked if score > 0]
        if active:
            lines.append("Main contributors to the fused result: " + ", ".join(active) + ".")

    for output in module_outputs:
        if output.status != "available":
            continue
        top_features = output.explanations.get("top_features", [])[:3]
        if top_features:
            names = [
                f"{item.get('feature')} ({item.get('direction', 'model effect')})"
                for item in top_features
            ]
            lines.append(f"For {output.modality}, important model features included: {', '.join(names)}.")
    return "Explanation summary: " + " ".join(lines) if lines else "Explanation summary: no detailed explanation is available yet."


def _suggestions(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> str:
    suggestions = [
        "Discuss these results with a clinician, especially if symptoms such as a persistent ulcer, pain, bleeding, or neck swelling are present.",
        "Ask whether an oral examination, biopsy, imaging, or pathology review is appropriate based on clinical context.",
    ]
    missing = [output.modality for output in module_outputs if output.status != "available"]
    if missing:
        suggestions.append("If available, adding missing information may improve review quality: " + ", ".join(missing) + ".")
    if fusion_output and fusion_output.warnings:
        suggestions.append("Review the warning messages before interpreting the score.")
    return "Suggested next steps: " + " ".join(suggestions)


def _availability(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> str:
    available = [output.modality for output in module_outputs if output.status == "available"]
    missing = [output.modality for output in module_outputs if output.status != "available"]
    if fusion_output:
        missing.extend(
            modality
            for modality, contribution in fusion_output.modality_contributions.items()
            if contribution == 0 and modality not in available and modality not in missing
        )
    return (
        "Available information: "
        + (", ".join(sorted(set(available))) if available else "none")
        + ". Missing or unused information: "
        + (", ".join(sorted(set(missing))) if missing else "none")
        + "."
    )


def _sources_used(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> list[str]:
    sources = [output.modality for output in module_outputs if output.status == "available"]
    if fusion_output:
        sources.append("fusion")
    return sorted(set(sources))


def _mentions(query: str, words: tuple[str, ...]) -> bool:
    return any(word in query for word in words)


def _is_greeting(query: str) -> bool:
    cleaned = query.strip().lower().strip("!.?, ")
    return cleaned in {"hi", "hey", "hello", "hola", "namaste", "yo", "hii", "hiii", "good morning", "good evening"}


def _chatty_response(
    patient_id: str,
    query: str,
    answer: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    intent: str,
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "query": query,
        "answer": answer,
        "disclaimer": DISCLAIMER,
        "sources_used": _sources_used(module_outputs, fusion_output),
        "suggested_questions": _suggested_questions(),
        "answer_mode": "rule_based",
        "llm_provider": None,
        "llm_model": None,
        "intent": intent,
    }


def _suggested_questions() -> list[str]:
    return [
        "What is my current risk status?",
        "Which information influenced the result most?",
        "What information is missing?",
        "What should I discuss with my doctor?",
    ]


def _patient_context(
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "disclaimer": DISCLAIMER,
        "patient_record_context": extra_context or {},
        "fusion": fusion_output.model_dump(by_alias=True) if fusion_output else None,
        "modalities": [
            {
                "modality": output.modality,
                "status": output.status,
                "mode": output.mode,
                "prediction": output.prediction,
                "confidence": output.confidence,
                "quality_flags": output.quality_flags,
                "warnings": output.warnings,
                "explanations": {
                    "method": output.explanations.get("method"),
                    "note": output.explanations.get("note"),
                    "top_features": output.explanations.get("top_features", [])[:8],
                },
            }
            for output in module_outputs
        ],
    }
