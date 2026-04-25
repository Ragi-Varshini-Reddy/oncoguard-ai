# OralCare-AI Impact Pitch

## Title

**OralCare-AI: Explainable Multimodal Decision Support for Early Oral Cancer Detection**

## One-Line Pitch

OralCare-AI turns fragmented oral cancer evidence into a doctor-reviewed, explainable, patient-understandable risk timeline.

## Problem Statement

Oral cancer is often detected late because the evidence needed for decision-making is scattered across multiple places:

- Intraoral photos may show visible lesions.
- Histopathology reports may arrive separately.
- Genomics data may reveal molecular risk.
- Clinical history may contain tobacco, alcohol, hygiene, lesion, and symptom risk factors.
- Patients may notice daily changes before the next hospital visit.

Most digital tools treat these signals separately. A single image model gives a score. A report stays as a PDF. A patient waits. A doctor has to mentally combine everything under time pressure.

The problem is not only prediction. The problem is fragmented, delayed, and poorly explained evidence.

## Need of the Problem Statement

Early oral cancer detection matters because small delays can change outcomes dramatically. Suspicious oral lesions can evolve through precancerous stages before becoming invasive disease. In many real workflows, the first signal may not come from a hospital-grade scan. It may come from a patient’s daily mouth photo, a clinical observation, a pathology report, or a molecular marker.

Doctors need a system that helps them:

- See all patient evidence in one place.
- Prioritize high-risk patients earlier.
- Understand why AI flagged a case.
- Validate the AI before results reach the patient.
- Track risk over time instead of seeing a one-time score.

Patients need a system that helps them:

- Understand their status in simple language.
- Upload daily intraoral images for early-change monitoring.
- Know when doctor review is needed.
- Download an approved report.
- Ask questions without being overwhelmed by medical jargon.

This is why OralCare-AI is built as a workflow, not just a model.

## Existing Work

Existing oral cancer AI work generally falls into five groups:

1. **Single-image classification models**  
   These classify oral lesion images as benign, suspicious, or cancer-like.

2. **Histopathology classifiers**  
   These detect OSCC-like patterns from H&E slide images or patches.

3. **Clinical risk calculators**  
   These estimate risk from age, tobacco, alcohol, lesions, symptoms, and other structured clinical features.

4. **Genomics classifiers**  
   These use gene expression or mutation panels to estimate cancer-related molecular risk.

5. **Medical chatbots**  
   These explain general health information, but often do not deeply integrate patient-specific multimodal evidence.

These systems are valuable, but most are narrow. They solve one technical slice of the problem, not the full clinical journey.

## Gaps in Existing Work

### Gap 1: Single-modality thinking

Many tools work on one input type. Oral cancer decision-making is naturally multimodal. Images, pathology, genomics, and clinical history each explain a different layer of risk.

### Gap 2: No missing-modality strategy

Real clinics rarely have every report at the same time. A useful system should still work when only one or two modalities are available and clearly explain what is missing.

### Gap 3: Weak doctor-in-the-loop workflow

Many AI demos go directly from model output to patient-facing result. In healthcare, that is risky. Doctors need to validate, edit, and approve the report.

### Gap 4: Limited longitudinal tracking

A one-time risk score misses the patient journey. Daily intraoral uploads can reveal change over time, which is important for early detection and follow-up.

### Gap 5: Poor explainability

A score alone does not build trust. Doctors need modality contribution, decision trace, warning messages, and what-if evidence.

### Gap 6: Chatbots are often generic

Generic medical chatbots may answer well in general, but they often do not know the patient’s actual doctor, reports, risk trend, approved result, or processed model evidence.

## Our Solution

OralCare-AI is a multimodal clinical decision-support platform for oral cancer risk review.

It combines:

- **Intraoral AI** for patient and lab-uploaded oral images.
- **Histopathology AI** for tissue-level evidence and heatmap support.
- **Clinical AI** for structured patient risk factors.
- **Genomics AI** for molecular risk signals.
- **Fusion AI** to combine available evidence even when some modalities are missing.
- **XAI** to explain modality contribution, decision trace, warnings, and what-if changes.
- **Doctor dashboard** for review, editing, validation, and report approval.
- **Patient dashboard** for status, daily uploads, risk trend, chatbot, appointments, and PDF download.
- **RAG chatbot** that answers from processed patient context instead of raw image files.

## Solution Architecture in Simple Words

Each modality produces a standard output:

- prediction
- confidence
- embedding
- explanation
- warnings
- quality flags

The fusion layer then asks:

- Which modalities are available?
- How confident is each model?
- Do the modalities agree or disagree?
- Which modality contributed most?
- What happens if one modality is removed?
- Is this high-risk enough to alert the doctor?

The doctor sees the answer as a reviewable clinical story, not a black-box score.

## Technical Differentiation

### 1. Missing-modality-aware fusion

OralCare-AI does not fail when genomics or pathology is missing. It uses the available evidence and clearly marks missing modalities.

### 2. Doctor-approved patient result

Patients do not just receive raw AI output. The doctor edits and approves the final report first.

### 3. Daily intraoral monitoring

Patients can upload daily oral images. This turns the platform from a one-time classifier into a longitudinal early-detection companion.

### 4. XAI for both doctors and patients

Doctors get technical explanation. Patients get simple explanation. Both are grounded in the same evidence.

### 5. RAG without raw image context

The chatbot uses processed summaries and vector-retrieved patient context. It does not send raw intraoral or histopathology images to the LLM.

## Prototype Metrics for Demo

These are internal prototype validation metrics for presentation and should be replaced with final locked test-set results before publication.

| Component | Prototype metric |
|---|---:|
| Intraoral image model AUROC | 0.862 |
| Histopathology model AUROC | 0.985 |
| Clinical model AUROC | 0.934 |
| Genomics model AUROC | 0.851 |
| Fusion model AUROC | 0.791 |
| Fusion sensitivity | 78.4% |
| Missing-modality support | Any subset of 4 modalities |
| Explainability coverage | Decision trace, contribution chart, warnings, what-if |

## Impact

OralCare-AI can help reduce the gap between first patient-observed change and doctor-reviewed action.

It improves:

- early suspicion detection
- doctor prioritization
- patient understanding
- report transparency
- longitudinal monitoring
- multimodal decision support

The biggest impact is that it makes oral cancer AI clinically usable: not just a prediction, but a workflow.

## 60-Second Pitch

"Oral cancer is often detected late because evidence is fragmented. A patient may have oral photos, the lab may have pathology, the clinician has risk history, and genomics may arrive separately. Most AI systems look at only one of these and output a score.

OralCare-AI brings all of this into one explainable workflow. We process intraoral images, histopathology, clinical data, and genomics into standard AI outputs. Our fusion layer combines available evidence, even when some modalities are missing, and explains the decision using contribution charts, decision trace, warnings, and what-if analysis.

Most importantly, this is doctor-in-the-loop. The doctor reviews, edits, and approves the report before the patient sees it. Patients can also upload daily intraoral images, track risk over time, download approved reports, book appointments, and ask a chatbot that answers from their own processed record.

OralCare-AI is not replacing doctors. It is giving doctors an explainable multimodal co-pilot and giving patients a simple, safe way to understand their care."

## 3-Minute Judge Pitch

"Our problem statement is oral cancer early detection. The challenge is not just that cancer is hard to predict. The real challenge is that the evidence is scattered. Intraoral images, histopathology slides, genomics, and clinical risk factors often exist in separate workflows. A doctor has to manually connect them, and patients often do not understand what the results mean.

Existing AI systems are usually single-modality. They classify one image, one slide, or one table. That is useful, but incomplete. Oral cancer risk is multimodal, longitudinal, and clinical. A one-time score is not enough.

Our solution is OralCare-AI, an explainable multimodal decision-support platform. The lab technician uploads modality-specific files. Each AI model produces a standard output: prediction, confidence, embedding, explanation, warnings, and quality flags. These outputs go into our evidence-weighted fusion layer.

The fusion model is missing-modality-aware. That matters because real clinics rarely have all reports ready at the same time. If histopathology is missing, the system still works and tells the doctor exactly what is missing. If genomics is present, it adds molecular evidence. If daily intraoral uploads are available, the system tracks risk over time.

The doctor dashboard is not just a viewer. It is a validation workspace. The doctor sees patient names, risk stats, uploaded documents, charts, decision trace, and XAI. The doctor can edit the report before approving it. Only after approval does the patient see the final status and download the PDF.

The patient dashboard is designed for clarity. Patients can upload daily intraoral images, see their trend, request appointments, and chat with OralCare-AI. The chatbot is grounded using RAG over processed patient context, not raw images. It answers simply and safely.

The impact is a complete early-detection workflow: multimodal AI, explainable fusion, doctor validation, patient communication, and longitudinal monitoring. That is what makes OralCare-AI different from a model demo. It is a clinical product prototype."

## Closing Line

"We are not building another cancer classifier. We are building the explainable workflow around it, so evidence becomes action faster."

