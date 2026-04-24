const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const storedUser = JSON.parse(localStorage.getItem("oralcare_user") || "null");
  const headers = new Headers(options.headers || {});
  if (storedUser?.user?.user_id) {
    headers.set("X-User-Id", storedUser.user.user_id);
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Request failed");
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response;
}

export function getHealth() {
  return request("/api/health");
}

export function getRoles() {
  return request("/api/users/roles");
}

export function getDemoUsers() {
  return request("/api/auth/demo-users");
}

export function demoLogin(userId) {
  return request("/api/auth/demo-login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
}

export function getMe() {
  return request("/api/me");
}

export function getDoctorPatients(doctorId) {
  return request(`/api/doctors/${encodeURIComponent(doctorId)}/patients`);
}

export function getPatient(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}`);
}

export function getPatientDocuments(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}/documents`);
}

export function getPatientModelRuns(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}/model-runs`);
}

export function getAppointments() {
  return request("/api/appointments");
}

export function getPatientRiskHistory(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}/risk-history`);
}

export function getPatientAlerts(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}/alerts`);
}

export function getAllAlerts() {
  return request("/api/alerts");
}

export function requestAppointment(patientId, payload) {
  return request(`/api/patients/${encodeURIComponent(patientId)}/appointments/request`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateAppointment(appointmentId, payload) {
  return request(`/api/appointments/${encodeURIComponent(appointmentId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function uploadPatientDocument(patientId, file, documentType = "genomics", notes = "") {
  const form = new FormData();
  form.append("document_file", file);
  form.append("document_type", documentType);
  form.append("notes", notes);
  return request(`/api/patients/${encodeURIComponent(patientId)}/documents`, { method: "POST", body: form });
}

export function getGenomicsSchema() {
  return request("/api/genomics/schema");
}

export function trainGenomics(file) {
  const form = new FormData();
  form.append("training_file", file);
  return request("/api/genomics/train", { method: "POST", body: form });
}

export function inferGenomicsFile(file, patientId) {
  const form = new FormData();
  form.append("genomic_file", file);
  const suffix = patientId ? `?patient_id=${encodeURIComponent(patientId)}` : "";
  return request(`/api/genomics/infer-file${suffix}`, { method: "POST", body: form });
}

export function inferFusion(patientId, moduleOutputs, disabledModalities = []) {
  return request("/api/fusion/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      module_outputs: moduleOutputs,
      disabled_modalities: disabledModalities,
    }),
  });
}

export function explainFusion(patientId, moduleOutputs, disabledModalities = []) {
  return request("/api/fusion/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      module_outputs: moduleOutputs,
      disabled_modalities: disabledModalities,
    }),
  });
}

export async function downloadPdf(patientId, moduleOutputs, fusionOutput) {
  const response = await request("/api/reports/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      module_outputs: moduleOutputs,
      fusion_output: fusionOutput,
    }),
  });
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${patientId}_oralcare_ai_report.pdf`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function askPatientQuery(patientId, query, moduleOutputs, fusionOutput) {
  return request("/api/patient/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      query,
      module_outputs: moduleOutputs,
      fusion_output: fusionOutput,
    }),
  });
}

export function sendPatientChat(patientId, sessionId, message, moduleOutputs, fusionOutput) {
  return request("/api/patient/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      session_id: sessionId,
      message,
      module_outputs: moduleOutputs,
      fusion_output: fusionOutput,
      use_llm: true,
    }),
  });
}
