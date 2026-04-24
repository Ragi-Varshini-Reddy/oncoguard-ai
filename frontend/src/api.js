const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
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

export function inferFusion(patientId, moduleOutputs) {
  return request("/api/fusion/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId, module_outputs: moduleOutputs }),
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
