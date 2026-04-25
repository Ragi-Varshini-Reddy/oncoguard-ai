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
    intent = _classify_intent(normalized)
    sections: list[str] = []

    if intent == "greeting":
        sections.append(_greeting_answer(context))
    elif intent == "doctor":
        sections.append(_doctor_answer(extra_context))
    elif intent == "daily_check":
        sections.append(_daily_check_answer(extra_context))
    elif intent == "capabilities":
        sections.append(_capabilities_answer(module_outputs, fusion_output, extra_context))
    elif intent == "cancer_prone":
        sections.append(_cancer_prone_answer(fusion_output, module_outputs, extra_context))
    elif intent == "suggestions":
        sections.append(_suggestions(module_outputs, fusion_output, extra_context))
    elif intent == "risk":
        sections.append(_risk_summary(fusion_output, module_outputs))
    elif intent == "xai":
        sections.append(_xai_summary(module_outputs, fusion_output))
    elif intent == "availability":
        sections.append(_availability(module_outputs, fusion_output))

    if len(sections) == 0:
        sections.append(_casual_default_answer(fusion_output, module_outputs, extra_context))

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
            f"Right now, your AI risk level is **{risk['class']}** ({risk['score']:.1%}). "
            "That means your reports should be reviewed carefully by your doctor. It does **not** mean you definitely have cancer."
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
    return "What the available reports show: " + "; ".join(parts) + "."


def _greeting_answer(context: dict[str, Any]) -> str:
    patient = context.get("patient_record_context", {}).get("patient_record") or {}
    name = patient.get("name")
    available = [item.get("modality") for item in context.get("modalities", []) if item.get("status") == "available"]
    intro = f"Hi {name}, I’m OralCare-AI." if name else "Hi, I’m OralCare-AI."
    if available:
        return f"{intro} I can help explain your {', '.join(available)} results, risk trend, daily uploads, and doctor-reviewed report."
    return f"{intro} I can help once your reports or daily intraoral images are processed."


def _xai_summary(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> str:
    lines: list[str] = []
    if fusion_output and fusion_output.modality_contributions:
        ranked = sorted(fusion_output.modality_contributions.items(), key=lambda item: item[1], reverse=True)
        active = [f"{modality} ({score:.1%})" for modality, score in ranked if score > 0]
        if active:
            lines.append("The result was mainly influenced by: " + ", ".join(active) + ".")

    for output in module_outputs:
        if output.status != "available":
            continue
        top_features = output.explanations.get("top_features", [])[:3]
        if top_features:
            names = [
                f"{item.get('feature')} ({item.get('direction', 'model effect')})"
                for item in top_features
            ]
            lines.append(f"In the {output.modality} report, important signals were: {', '.join(names)}.")
        recommendations = output.explanations.get("recommendations", [])
        if recommendations:
            lines.append(f"Suggested actions: {'; '.join(str(item) for item in recommendations)}.")
    return "Simple explanation: " + " ".join(lines) if lines else "Simple explanation: no detailed explanation is available yet."


def _classify_intent(query: str) -> str:
    if _is_greeting(query):
        return "greeting"
    if _mentions(query, ("doctor", "clinician", "physician", "who is treating", "assigned")):
        return "doctor"
    if _mentions(query, ("daily intraoral", "daily check", "intraoral ai check", "upload image", "daily upload", "mouth photo")):
        return "daily_check"
    if _mentions(query, ("what can you do", "help me", "how can you help", "your role", "features")):
        return "capabilities"
    if _mentions(query, ("prone", "oral cancer", "do i have cancer", "am i cancer", "cancer now")):
        return "cancer_prone"
    if _mentions(query, ("suggest", "advice", "next", "should", "recommend", "do now", "steps")):
        return "suggestions"
    if _mentions(query, ("risk", "status", "health", "score", "condition", "how am i")):
        return "risk"
    if _mentions(query, ("why", "explain", "reason", "because", "feature", "gene", "xai")):
        return "xai"
    if _mentions(query, ("missing", "available", "provided", "modality", "data")):
        return "availability"
    return "general"


def _doctor_answer(extra_context: dict[str, Any] | None) -> str:
    doctor = (extra_context or {}).get("doctor_details") or {}
    name = doctor.get("name")
    if not name:
        return "I do not see an assigned doctor in your current record context. Please check your dashboard or ask the clinic team to link a doctor."
    specialty = doctor.get("specialty")
    clinic = doctor.get("clinic_name")
    location = doctor.get("clinic_location")
    details = []
    if specialty:
        details.append(str(specialty))
    if clinic:
        details.append(str(clinic))
    if location:
        details.append(str(location))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Your doctor is **{name}**{suffix}. That comes from your linked patient record."


def _daily_check_answer(extra_context: dict[str, Any] | None) -> str:
    documents = (extra_context or {}).get("documents") or []
    intraoral_docs = [doc for doc in documents if doc.get("document_type") == "intraoral"]
    if not intraoral_docs:
        return "I do not see daily intraoral uploads in your current record yet. When you submit one, it is stored by date and can update your risk trend after AI processing."
    latest = sorted(intraoral_docs, key=lambda doc: str(doc.get("created_at") or ""), reverse=True)[0]
    return (
        f"I see **{len(intraoral_docs)}** recent intraoral upload(s) in your record context. "
        f"The latest one is **{latest.get('filename', 'an intraoral image')}** from {str(latest.get('created_at', 'the latest date'))[:10]}. "
        "These uploads are used for date-wise tracking and early-change monitoring."
    )


def _capabilities_answer(
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    extra_context: dict[str, Any] | None,
) -> str:
    items = []
    if fusion_output:
        items.append("explain your current AI risk and confidence")
        items.append("summarize what influenced the fused result")
    available = [output.modality for output in module_outputs if output.status == "available"]
    if available:
        items.append("walk through your processed " + ", ".join(sorted(set(available))) + " results")
    docs = (extra_context or {}).get("documents") or []
    if any(doc.get("document_type") == "intraoral" for doc in docs):
        items.append("explain your daily intraoral image uploads and date-wise trend")
    if (extra_context or {}).get("doctor_details"):
        items.append("tell you who your assigned doctor is")
    if not items:
        items.append("explain your reports after they are processed")
    return "I can help with " + "; ".join(items[:5]) + ". I will stay within your record and avoid guessing."


def _cancer_prone_answer(
    fusion_output: FusionOutput | None,
    module_outputs: list[ModuleOutput],
    extra_context: dict[str, Any] | None,
) -> str:
    if not fusion_output:
        return "I do not have enough processed report data to estimate your current risk yet. Uploads and doctor review are needed before interpreting anything."
    risk = fusion_output.risk.model_dump(by_alias=True)
    doctor = ((extra_context or {}).get("doctor_details") or {}).get("name") or "your doctor"
    return (
        f"Your current AI result is **{risk['class']} risk** ({risk['score']:.1%}). "
        "So the system is flagging your record for careful clinical review, but this is **not a confirmed cancer diagnosis**. "
        f"Please review this with {doctor}, who can decide what clinical step is appropriate."
    )


def _casual_default_answer(
    fusion_output: FusionOutput | None,
    module_outputs: list[ModuleOutput],
    extra_context: dict[str, Any] | None,
) -> str:
    doctor = ((extra_context or {}).get("doctor_details") or {}).get("name")
    available = [output.modality for output in module_outputs if output.status == "available"]
    parts = []
    if fusion_output:
        risk = fusion_output.risk.model_dump(by_alias=True)
        parts.append(f"I see a current AI risk result: **{risk['class']}** ({risk['score']:.1%}).")
    elif available:
        parts.append("I see processed data for: " + ", ".join(sorted(set(available))) + ".")
    else:
        parts.append("I do not see processed report data in this chat context yet.")
    if doctor:
        parts.append(f"Your assigned doctor in the record is {doctor}.")
    parts.append("Ask me a specific question, and I’ll answer from your record.")
    return " ".join(parts)


def _feature_importance(item: dict[str, Any]) -> float:
    raw_value = item.get("importance_score", item.get("shap_value", 0.0))
    try:
        return abs(float(raw_value))
    except (TypeError, ValueError):
        return 0.0


def _top_feature_importance(top_features: list[dict[str, Any]], feature: str) -> float:
    for item in top_features:
        if item.get("feature") == feature:
            return _feature_importance(item)
    return 0.0


def _current_risk_percent(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> int | None:
    if fusion_output:
        try:
            return round(float(fusion_output.risk.score) * 100)
        except (TypeError, ValueError, AttributeError):
            pass
    for output in module_outputs:
        if output.status != "available":
            continue
        risk_score = (output.prediction or {}).get("risk_score")
        if isinstance(risk_score, (int, float)):
            return round(float(risk_score) * 100)
    return None


def _shap_adjusted_risk(current_risk_percent: int | None, feature_importance: float, total_importance: float) -> int | None:
    if current_risk_percent is None:
        return None
    if feature_importance <= 0:
        return current_risk_percent
    denominator = total_importance if total_importance > 0 else feature_importance
    share = feature_importance / denominator if denominator > 0 else 0.0
    estimated_drop = max(3.0, float(current_risk_percent) * min(0.25, share * 0.35))
    return max(0, round(float(current_risk_percent) - estimated_drop))


def _risk_guidance(module_outputs: list[ModuleOutput], fusion_output: FusionOutput | None) -> dict[str, Any]:
    current_risk_percent = _current_risk_percent(module_outputs, fusion_output)
    clinical_output = next(
        (
            output
            for output in module_outputs
            if output.status == "available" and output.modality == "clinical"
        ),
        None,
    )
    healthy_habit = None
    if not clinical_output:
        return {
            "current_risk_percent": current_risk_percent,
            "habit_change": None,
            "healthy_habit": healthy_habit,
        }

    feature_values = clinical_output.explanations.get("feature_values", {}) or {}
    top_features = clinical_output.explanations.get("top_features", []) or []
    active_habit = None
    for feature in ("tobacco_use", "alcohol_use"):
        if feature_values.get(feature):
            active_habit = feature
            break
    habit_change = None
    if active_habit:
        if feature_values.get("tobacco_use") and feature_values.get("alcohol_use"):
            active_habit = max(
                ("tobacco_use", "alcohol_use"),
                key=lambda feature: _top_feature_importance(top_features, feature),
            )
        feature_importance = _top_feature_importance(top_features, active_habit)
        total_importance = sum(_feature_importance(item) for item in top_features if _feature_importance(item) > 0)
        adjusted_risk = _shap_adjusted_risk(current_risk_percent, feature_importance, total_importance)
        habit_name = "Quit smoking" if active_habit == "tobacco_use" else "Reduce or stop alcohol"
        risk_label = f"Current risk {current_risk_percent}%" if current_risk_percent is not None else "Current risk unavailable"
        if adjusted_risk is None:
            estimate_label = "illustrative SHAP estimate unavailable"
        else:
            estimate_label = f"illustrative SHAP-adjusted risk {adjusted_risk}%"
        habit_change = {
            "feature": active_habit,
            "title": habit_name,
            "metric": f"{risk_label} -> {estimate_label}",
            "detail": f"{habit_name.lower()} is one of the strongest positive SHAP drivers in the latest clinical explanation.",
        }

    return {
        "current_risk_percent": current_risk_percent,
        "habit_change": habit_change,
        "healthy_habit": healthy_habit,
    }


def _suggestions(
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput | None,
    extra_context: dict[str, Any] | None = None,
) -> str:
    guidance = _risk_guidance(module_outputs, fusion_output)
    suggestions = []
    habit_change = guidance.get("habit_change")
    if habit_change:
        suggestions.append(
            f"**{habit_change['title']}**: {habit_change['metric']}."
        )
    clinical_recommendations = [
        recommendation
        for output in module_outputs
        for recommendation in output.explanations.get("recommendations", [])
        if output.status == "available"
    ]
    if clinical_recommendations:
        suggestions.append("**Model-guided lifestyle note**: " + " ".join(str(item) for item in clinical_recommendations[:2]))
    if guidance.get("current_risk_percent") is not None:
        suggestions.append(f"**Current AI risk**: {guidance['current_risk_percent']}%.")
    doctor_name = ((extra_context or {}).get("doctor_details") or {}).get("name")
    if doctor_name:
        suggestions.append(f"**Doctor review**: discuss this with {doctor_name}, using your latest uploads and AI trend.")
    else:
        suggestions.append("**Doctor review**: discuss this with your assigned clinician.")
    docs = (extra_context or {}).get("documents") or []
    if any(doc.get("document_type") == "intraoral" for doc in docs):
        suggestions.append("**Track changes**: keep using your dated intraoral uploads so your trend stays current.")
    missing = [output.modality for output in module_outputs if output.status != "available"]
    if missing:
        suggestions.append("More information may make the review clearer: " + ", ".join(missing) + ".")
    if fusion_output and fusion_output.warnings:
        suggestions.append("Ask your doctor about the warning messages before relying on the score.")
    if not suggestions:
        return "I do not see enough processed context to suggest specific next steps yet."
    return "Based on your current record:\n- " + "\n- ".join(suggestions[:5])


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
        "Information used: "
        + (", ".join(sorted(set(available))) if available else "none")
        + ". Missing or not used yet: "
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
        "risk_guidance": _risk_guidance(module_outputs, fusion_output),
        "xai_context": {
            "plain_language_summary": _xai_summary(module_outputs, fusion_output),
            "fusion_contributions": fusion_output.modality_contributions if fusion_output else {},
            "decision_trace": fusion_output.decision_trace if fusion_output else [],
            "modality_evidence": fusion_output.modality_evidence if fusion_output else {},
        },
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
                    "feature_values": output.explanations.get("feature_values", {}),
                    "recommendations": output.explanations.get("recommendations", []),
                },
            }
            for output in module_outputs
        ],
    }
