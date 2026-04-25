import React, { useEffect, useMemo, useState, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart as RechartsPie, Pie, Cell, Legend, Sector
} from "recharts";
import {
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Download,
  Eye,
  FileText,
  PieChart,
  Save,
  ShieldCheck,
  Stethoscope,
  TrendingUp,
} from "lucide-react";
import {
  approvePatientReport,
  downloadPdf,
  getDoctorPatients,
  getPatient,
  getPatientDocumentBlob,
  getPatientRiskHistory,
  createPatient,
} from "../services/api";
import AppointmentsPanel from "./AppointmentsPanel";

export default function DoctorDashboard({ user, view = "dashboard" }) {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [riskHistory, setRiskHistory] = useState([]);
  const [showRisk, setShowRisk] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [reportText, setReportText] = useState("");
  const [doctorNotes, setDoctorNotes] = useState("");
  const [approvalState, setApprovalState] = useState("");
  const [documentPreview, setDocumentPreview] = useState(null);
  const [comparePreview, setComparePreview] = useState(null); // {original, heatmap}
  const [documentModal, setDocumentModal] = useState(null); // {mode: "single" | "compare", title, content}
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: "", age: "", sex: "male", summary: "", email: "", phone: "" });
  const [addPatientStatus, setAddPatientStatus] = useState("");
  const [patientSearch, setPatientSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");
  const [activePatientTab, setActivePatientTab] = useState("overview");
  const doctorId = user?.profile?.doctor_id || "DR-RAO";

  const riskChartRef = useRef(null);
  const traceListRef = useRef(null);

  useEffect(() => {
    loadPatients();
  }, [doctorId]);

  useEffect(() => {
    if (view === "reports" && patients.length > 0 && !selectedPatient) {
      handleSelectPatient(patients[0]);
    }
  }, [view, patients, selectedPatient]);

  useEffect(() => {
    return () => {
      if (documentPreview?.url) URL.revokeObjectURL(documentPreview.url);
      if (comparePreview?.original?.url) URL.revokeObjectURL(comparePreview.original.url);
      if (comparePreview?.heatmap?.url) URL.revokeObjectURL(comparePreview.heatmap.url);
    };
  }, [documentPreview, comparePreview]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setDocumentModal(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  async function loadPatients() {
    try {
      const data = await getDoctorPatients(doctorId);
      setPatients(data.patients);
    } catch (err) {
      console.error("Failed to load doctor's patients", err);
    } finally {
      setLoading(false);
    }
  }

  async function handlePreviewDocument(patientId, doc) {
    setComparePreview(null);
    try {
      const blob = await getPatientDocumentBlob(patientId, doc.document_id);
      if (documentPreview?.url) URL.revokeObjectURL(documentPreview.url);
      const preview = {
        url: URL.createObjectURL(blob),
        filename: doc.filename,
        type: blob.type || guessMime(doc.filename),
      };
      setDocumentPreview(preview);
      setDocumentModal({
        mode: "single",
        title: doc.filename,
        content: preview,
      });
    } catch (err) {
      const errorPreview = { error: err.message, filename: doc.filename };
      setDocumentPreview(errorPreview);
      setDocumentModal({
        mode: "single",
        title: doc.filename,
        content: errorPreview,
      });
    }
  }

  async function handleSelectPatient(p) {
    try {
      const data = await getPatient(p.patient_id);
      setSelectedPatient(data);
      setRiskHistory([]);
      setShowRisk(false);
      setShowTrace(false);
      setApprovalState("");
      setDoctorNotes(data.latest_approval?.doctor_notes || "");
      setReportText(data.latest_approval?.report_text || buildReportDraft(data));
      setActivePatientTab("overview");
    } catch (err) {
      console.error("Failed to load patient detail", err);
    }
  }

  async function refreshSelected() {
    if (!selectedPatient?.patient?.patient_id) return;
    await handleSelectPatient(selectedPatient.patient);
  }

  async function handleCreatePatient(e) {
    e.preventDefault();
    setAddPatientStatus("Creating...");
    try {
      await createPatient({
        ...newPatient,
        age: newPatient.age ? parseInt(newPatient.age, 10) : undefined
      });
      setAddPatientStatus("");
      setShowAddPatient(false);
      setNewPatient({ name: "", age: "", sex: "male", summary: "", email: "", phone: "" });
      loadPatients();
    } catch (err) {
      setAddPatientStatus(err.message || "Failed to create patient");
    }
  }

  async function handleRiskHistory() {
    if (!selectedPatient) return;
    const data = await getPatientRiskHistory(selectedPatient.patient.patient_id);
    setRiskHistory(data.risk_history || []);
    setShowRisk(true);
    setTimeout(() => {
      riskChartRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  }

  async function handleCompareDocs(patientId, originalDoc, heatmapDoc) {
    setDocumentPreview(null);
    try {
      const [origBlob, heatBlob] = await Promise.all([
        getPatientDocumentBlob(patientId, originalDoc.document_id),
        getPatientDocumentBlob(patientId, heatmapDoc.document_id),
      ]);
      const original = { url: URL.createObjectURL(origBlob), filename: originalDoc.filename };
      const heatmap = { url: URL.createObjectURL(heatBlob), filename: heatmapDoc.filename };
      setComparePreview({
        original,
        heatmap,
      });
      setDocumentModal({
        mode: "compare",
        title: `${originalDoc.filename} vs ${heatmapDoc.filename}`,
        content: {
          original,
          heatmap,
        },
      });
    } catch (err) {
      const errorPreview = { error: err.message, filename: "comparison" };
      setDocumentPreview(errorPreview);
      setDocumentModal({
        mode: "single",
        title: "comparison",
        content: errorPreview,
      });
    }
  }

  async function handleRegistryDocumentAction(patientSummary, action) {
    try {
      const patientData = await getPatient(patientSummary.patient_id);
      setSelectedPatient(patientData);
      setRiskHistory([]);
      setShowRisk(false);
      setShowTrace(false);
      setApprovalState("");
      setDoctorNotes(patientData.latest_approval?.doctor_notes || "");
      setReportText(patientData.latest_approval?.report_text || buildReportDraft(patientData));

      const documents = patientData.documents || [];
      const heatmapDoc = documents.find((doc) => doc.filename?.includes("_heatmap") || doc.notes?.includes("Grad-CAM"));
      const originalDoc = heatmapDoc
        ? documents.find((doc) => !doc.filename?.includes("_heatmap") && heatmapDoc.filename?.startsWith(doc.filename?.replace(/\.[^.]+$/, "")))
        : documents[0];

      if (action === "compare" && heatmapDoc && originalDoc) {
        await handleCompareDocs(patientData.patient.patient_id, originalDoc, heatmapDoc);
        return;
      }

      if (documents.length > 0) {
        await handlePreviewDocument(patientData.patient.patient_id, documents[0]);
      }
    } catch (err) {
      console.error("Failed to open patient documents", err);
    }
  }

  async function handleApproval(status) {
    if (!selectedPatient || !reportText.trim()) return;
    setApprovalState("Saving review...");
    try {
      await approvePatientReport(selectedPatient.patient.patient_id, {
        report_text: reportText,
        approval_status: status,
        doctor_notes: doctorNotes || null,
      });
      setApprovalState(status === "approved" ? "Approved. Patient dashboard is updated." : "Saved for follow-up.");
      await refreshSelected();
    } catch (err) {
      setApprovalState(err.message);
    }
  }

  const latestRun = selectedPatient?.latest_model_run;
  const fusion = latestRun?.fusion_output;
  const contributionRows = useMemo(() => {
    const contributions = fusion?.modality_contributions || {};
    return Object.entries(contributions)
      .filter(([, value]) => Number(value) > 0)
      .sort((a, b) => b[1] - a[1]);
  }, [fusion]);
  const xaiRows = useMemo(() => buildXaiRows(latestRun, fusion), [latestRun, fusion]);
  const filteredPatients = useMemo(() => {
    const query = patientSearch.trim().toLowerCase();
    return patients.filter((patient) => {
      const displayName = patient.name || patient.patient_id;
      const matchesSearch = !query
        || displayName.toLowerCase().includes(query)
        || String(patient.patient_id || "").toLowerCase().includes(query);
      const level = String(patient.latest_risk_level || "pending").toLowerCase();
      const matchesRisk = riskFilter === "all" || level === riskFilter;
      return matchesSearch && matchesRisk;
    });
  }, [patients, patientSearch, riskFilter]);
  const doctorStats = useMemo(() => {
    const highRisk = patients.filter((patient) => patient.latest_risk_level === "high").length;
    const pending = patients.filter((patient) => !patient.latest_risk_level).length;
    const reviewed = patients.length - pending;
    const averageRisk = reviewed
      ? patients.reduce((sum, patient) => sum + Number(patient.latest_risk_score || 0), 0) / reviewed
      : 0;
    return { total: patients.length, highRisk, pending, averageRisk };
  }, [patients]);

  return (
    <div className={selectedPatient ? `dashboard doctor-dashboard patient-selected tab-${activePatientTab}` : "dashboard doctor-dashboard"}>
      <header className="dashboard-header">
        <div className="header-info">
          <h1>Clinical Command Center</h1>
          <p>{view === "reports" ? "Edit, validate, approve, and generate patient reports" : "Oncology decision-support and patient monitoring"}</p>
        </div>
        {selectedPatient && (
          <button className="secondary" onClick={() => setSelectedPatient(null)}>
            <ClipboardList size={17} /> Change Patient
          </button>
        )}
      </header>

      <div className="dashboard-grid">
        {!selectedPatient && (
          <aside className="patient-sidebar panel doctor-registry-panel">
            <div className="panel-head" style={{ justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <ClipboardList size={20} />
                <h2>Patient Registry</h2>
              </div>
              <button className="secondary compact" onClick={() => setShowAddPatient(true)}>+ New Patient</button>
            </div>
            <div className="doctor-stats-grid">
              <StatCard label="Patients" value={String(doctorStats.total)} color="blue" />
              <StatCard label="High Risk" value={String(doctorStats.highRisk)} color="red" />
              <StatCard label="Pending" value={String(doctorStats.pending)} color="blue" />
              <StatCard label="Avg Risk" value={formatPercent(doctorStats.averageRisk)} color="red" />
            </div>
            <div className="registry-toolbar">
              <input
                value={patientSearch}
                onChange={(event) => setPatientSearch(event.target.value)}
                placeholder="Search patient name or ID"
              />
              <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
                <option value="all">All risk levels</option>
                <option value="high">High risk</option>
                <option value="medium">Medium risk</option>
                <option value="low">Low risk</option>
                <option value="pending">Pending</option>
              </select>
            </div>
            <div className="patient-list">
              {filteredPatients.map(p => (
                <div
                  key={p.patient_id}
                  className="patient-card"
                  onClick={() => handleSelectPatient(p)}
                >
                  <div className="card-top">
                    <strong>{p.name || p.patient_id}</strong>
                    <span className={`badge risk-${p.latest_risk_level || "pending"}`}>{p.latest_risk_level || "Pending"}</span>
                  </div>
                  <div className="patient-risk-bar">
                    <div style={{ width: `${Math.max(2, Number(p.latest_risk_score || 0) * 100)}%` }} />
                  </div>
                  <div className="patient-card-meta">
                    <span>{p.patient_id} · {p.sex}, {p.age}y</span>
                    <small>{p.document_count || 0} documents · {p.heatmap_count || 0} heatmaps</small>
                  </div>
                  <div className="patient-card-actions">
                    <button
                      className="secondary compact"
                      disabled={!p.document_count}
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRegistryDocumentAction(p, "view");
                      }}
                    >
                      <Eye size={16} /> View Images
                    </button>
                    <button
                      className="secondary compact accent"
                      disabled={!p.heatmap_count}
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRegistryDocumentAction(p, "compare");
                      }}
                    >
                      <Eye size={16} /> Compare
                    </button>
                  </div>
                </div>
              ))}
              {!loading && filteredPatients.length === 0 && <p className="muted">No matching patients found.</p>}
            </div>
          </aside>
        )}

        <main className="clinical-workspace">
          {!selectedPatient ? (
            <div className="empty-state panel">
              <Stethoscope size={48} />
              <h3>Select a Patient Record</h3>
              <p>Review uploaded documents, diagnostic evidence, fusion scores, and approvals.</p>
            </div>
          ) : (
            <div className="detail-container">
              <section className="patient-hero panel">
                <div className="hero-content">
                  <div className="patient-bio">
                    <h2>{selectedPatient.patient.name || selectedPatient.patient.patient_id}</h2>
                    <p className="summary">{selectedPatient.patient.summary}</p>
                    <div className="tags">
                      <span>ID: {selectedPatient.patient.patient_id}</span>
                      <span>Age: {selectedPatient.patient.age}</span>
                      <span>Sex: {selectedPatient.patient.sex}</span>
                      <span>Tech: {selectedPatient.technician?.name || "Unassigned"}</span>
                      <span>Review: {selectedPatient.latest_approval?.approval_status || "pending"}</span>
                    </div>
                  </div>
                  <div className="quick-stats">
                    <StatCard label="Fused Risk" value={formatPercent(fusion?.risk?.score)} color="red" />
                    <StatCard label="Confidence" value={formatPercent(fusion?.confidence)} color="blue" />
                  </div>
                </div>
              </section>
              <nav className="patient-tabs" aria-label="Patient chart sections">
                {[
                  ["overview", "Overview"],
                  ["report", "Report"],
                  ["documents", "Documents"],
                  ["risk", "Risk"],
                  ["trace", "Trace"],
                  ["appointments", "Appointments"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={activePatientTab === id ? "active" : ""}
                    onClick={() => {
                      setActivePatientTab(id);
                      if (id === "risk" && !showRisk) handleRiskHistory();
                      if (id === "trace") setShowTrace(true);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </nav>

              <div className="workspace-grid">
                {activePatientTab === "overview" && <section className="modality-outputs panel">
                  <div className="panel-head">
                    <BarChart3 size={18} />
                    <h3>Diagnostic Evidence</h3>
                  </div>
                  <div className="evidence-list">
                    {latestRun?.module_outputs?.map(m => (
                      <div key={m.modality} className="modality-card">
                        <div className="modality-header">
                          <strong>{m.modality.toUpperCase()}</strong>
                          <span className="status-badge available">Available</span>
                        </div>
                        <div className="modality-body">
                          <p>Prediction: <strong>{m.prediction?.diagnosis_class || m.prediction?.risk_class || "Model signal"}</strong></p>
                          <p>Confidence: {formatPercent(m.confidence)}</p>
                        </div>
                      </div>
                    ))}
                    {!latestRun?.module_outputs?.length && <p className="muted">No technician uploads found for this patient yet.</p>}
                  </div>
                </section>}

                {activePatientTab === "overview" && <section className="chart-panel panel">
                  <div className="panel-head">
                    <PieChart size={18} />
                    <h3>Fusion Proportions</h3>
                  </div>
                  <ContributionChart rows={contributionRows} />
                </section>}

                {activePatientTab === "overview" && <section className="xai-panel panel">
                  <div className="panel-head">
                    <ShieldCheck size={18} />
                    <h3>XAI Evidence</h3>
                  </div>
                  <XaiPanel rows={xaiRows} decisionTrace={fusion?.decision_trace || []} />
                </section>}

                {activePatientTab === "report" && <section className="report-actions panel">
                  <div className="panel-head">
                    <FileText size={18} />
                    <h3>Clinical Reporting</h3>
                  </div>
                  <div className="action-list">
                    <button
                      className="primary"
                      disabled={!latestRun}
                      onClick={() => downloadPdf(selectedPatient.patient.patient_id, latestRun.module_outputs, latestRun.fusion_output, reportText, doctorNotes)}
                    >
                      <Download size={17} /> Download Styled PDF Report
                    </button>
                    <button className="secondary" disabled={!latestRun} onClick={handleRiskHistory}>
                      <TrendingUp size={17} /> View Longitudinal Risk
                    </button>
                    <button className="secondary" disabled={!latestRun} onClick={() => {
                      setShowTrace(value => {
                        if (!value) {
                          setTimeout(() => traceListRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
                        }
                        return !value;
                      });
                    }}>
                      <ShieldCheck size={17} /> Validate Decision Trace
                    </button>
                  </div>
                  <div className="inline-report-editor">
                    <label>
                      Edit report before generating or approving
                      <textarea value={reportText} onChange={event => setReportText(event.target.value)} />
                    </label>
                    <label>
                      Doctor notes
                      <textarea value={doctorNotes} onChange={event => setDoctorNotes(event.target.value)} />
                    </label>
                    {approvalState && <p className="muted">{approvalState}</p>}
                    <div className="approval-actions">
                      <button className="secondary" disabled={!latestRun || !reportText.trim()} onClick={() => handleApproval("draft")}>
                        <Save size={17} /> Save Draft
                      </button>
                      <button className="primary" disabled={!latestRun || !reportText.trim()} onClick={() => handleApproval("approved")}>
                        <CheckCircle2 size={17} /> Approve Results
                      </button>
                    </div>
                  </div>
                </section>}

                {activePatientTab === "documents" && <section className="documents-panel panel">
                  <div className="panel-head">
                    <FileText size={18} />
                    <h3>Uploaded Documents and Images</h3>
                  </div>
                  <div className="document-grid">
                    <div className="document-list">
                      <div className="document-list-header">
                        <span>File</span>
                        <span>Type</span>
                        <span>Actions</span>
                      </div>
                      {(selectedPatient.documents || []).map(doc => {
                        const isHeatmap = doc.filename?.includes("_heatmap") || doc.notes?.includes("Grad-CAM");
                        // Find paired original for heatmaps
                        const pairedOriginal = isHeatmap
                          ? (selectedPatient.documents || []).find(d =>
                              !d.filename?.includes("_heatmap") &&
                              doc.filename?.startsWith(d.filename?.replace(/\.[^.]+$/, ""))
                            )
                          : null;
                        return (
                          <div key={doc.document_id} className={`document-row ${isHeatmap ? "heatmap-row" : ""}`}>
                            <div>
                              <strong>
                                {isHeatmap && <span className="heatmap-badge">🔬 Heatmap</span>}
                                {doc.filename}
                              </strong>
                              <span>{doc.document_type} · {new Date(doc.created_at).toLocaleString()}</span>
                              {doc.notes && <small>{doc.notes}</small>}
                            </div>
                            <div className="doc-actions">
                              <button className="secondary compact" onClick={() => handlePreviewDocument(selectedPatient.patient.patient_id, doc)}>
                                <Eye size={16} /> View
                              </button>
                              {isHeatmap && pairedOriginal && (
                                <button className="secondary compact accent" onClick={() => handleCompareDocs(selectedPatient.patient.patient_id, pairedOriginal, doc)}>
                                  <Eye size={16} /> Compare
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                      {(selectedPatient.documents || []).length === 0 && <p className="muted">No uploaded documents yet.</p>}
                    </div>
                  </div>
                </section>}

                {activePatientTab === "trace" && showTrace && (
                  <section className="wide-panel panel" ref={traceListRef}>
                    <div className="panel-head">
                      <ShieldCheck size={18} />
                      <h3>Validate Decision Trace</h3>
                    </div>
                    <div className="validation-grid">
                      <div className="trace-list">
                        {(fusion?.decision_trace || []).map((item, index) => (
                          <div key={`${item}-${index}`} className="trace-item">
                            <span>{index + 1}</span>
                            <p>{item}</p>
                          </div>
                        ))}
                        {!(fusion?.decision_trace || []).length && <p className="muted">No decision trace is available yet.</p>}
                      </div>
                      <div className="trace-summary">
                        <strong>Report validation</strong>
                        <p>Use the report editor in Clinical Reporting to revise the report, save a draft, approve results, or generate the PDF.</p>
                      </div>
                    </div>
                  </section>
                )}

                {activePatientTab === "risk" && showRisk && (
                  <section className="wide-panel panel" ref={riskChartRef}>
                    <div className="panel-head">
                      <TrendingUp size={18} />
                      <h3>Longitudinal Risk</h3>
                    </div>
                    <RiskTrendChart rows={riskHistory} />
                  </section>
                )}

                {activePatientTab === "appointments" && <section className="wide-panel">
                  <AppointmentsPanel user={user} />
                </section>}
              </div>
            </div>
          )}
        </main>
      </div>

      {showAddPatient && (
        <div className="modal-overlay">
          <div className="modal-content panel" style={{ maxWidth: "400px" }}>
            <div className="panel-head">
              <h2>Add New Patient</h2>
            </div>
            <form className="clinical-form" onSubmit={handleCreatePatient} style={{ display: "flex", flexDirection: "column" }}>
              <label>Name
                <input required value={newPatient.name} onChange={e => setNewPatient(p => ({...p, name: e.target.value}))} />
              </label>
              <label>Email
                <input type="email" value={newPatient.email} onChange={e => setNewPatient(p => ({...p, email: e.target.value}))} />
              </label>
              <label>Phone
                <input value={newPatient.phone} onChange={e => setNewPatient(p => ({...p, phone: e.target.value}))} />
              </label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <label>Age
                  <input type="number" min="0" value={newPatient.age} onChange={e => setNewPatient(p => ({...p, age: e.target.value}))} />
                </label>
                <label>Sex
                  <select value={newPatient.sex} onChange={e => setNewPatient(p => ({...p, sex: e.target.value}))}>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </label>
              </div>
              <label>Medical Summary
                <textarea rows="3" value={newPatient.summary} onChange={e => setNewPatient(p => ({...p, summary: e.target.value}))} style={{ width: "100%", padding: "8px", borderRadius: "8px", border: "1px solid #cfd9e2" }} />
              </label>
              {addPatientStatus && (
                <div className={`status-alert ${addPatientStatus === "Creating..." ? "info" : "error"}`}>
                  {addPatientStatus}
                </div>
              )}
              <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
                <button type="button" className="secondary" onClick={() => setShowAddPatient(false)}>Cancel</button>
                <button type="submit" className="primary">Create Patient</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {documentModal && (
        <div className="modal-overlay" onClick={() => setDocumentModal(null)}>
          <div className={`modal-content panel document-modal ${documentModal.mode === "compare" ? "document-modal-wide" : ""}`} onClick={(event) => event.stopPropagation()}>
            <div className="panel-head modal-header">
              <div>
                <h2>{documentModal.mode === "compare" ? "Compare Documents" : "View Document"}</h2>
                <p className="muted">{documentModal.title}</p>
              </div>
              <button className="close-btn" onClick={() => setDocumentModal(null)}>×</button>
            </div>
            {documentModal.mode === "compare" ? (
              <div className="compare-preview">
                <div className="compare-slot">
                  <p className="compare-label">Original</p>
                  <img src={documentModal.content.original.url} alt={documentModal.content.original.filename} />
                  <small>{documentModal.content.original.filename}</small>
                </div>
                <div className="compare-divider">
                  <span>vs</span>
                </div>
                <div className="compare-slot heatmap-slot">
                  <p className="compare-label">Grad-CAM Heatmap</p>
                  <img src={documentModal.content.heatmap.url} alt={documentModal.content.heatmap.filename} />
                  <small>{documentModal.content.heatmap.filename}</small>
                </div>
              </div>
            ) : (
              <DocumentPreview preview={documentModal.content} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div className={`stat-card ${color}`}>
      <span>{label}</span>
      <strong>{value || "N/A"}</strong>
    </div>
  );
}

const PIE_COLORS = ["#e53e3e", "#3182ce", "#38a169", "#805ad5", "#dd6b20", "#d69e2e"];

function ContributionChart({ rows }) {
  const [activeIndex, setActiveIndex] = useState(null);
  if (!rows.length) return <p className="muted">Fusion proportions will appear after reports are processed.</p>;

  const total = rows.reduce((sum, [, value]) => sum + Number(value || 0), 0) || 1;
  const data = rows.map(([name, value]) => ({
    name: prettyName(name),
    value: Math.round((Number(value) / total) * 100),
    raw: Number(value),
  }));

  const renderActiveShape = (props) => {
    const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent } = props;
    return (
      <g>
        <text x={cx} y={cy - 12} textAnchor="middle" fill="#1a202c" style={{ fontWeight: 700, fontSize: 18 }}>
          {(percent * 100).toFixed(0)}%
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="#718096" style={{ fontSize: 12 }}>
          {payload.name}
        </text>
        <Sector cx={cx} cy={cy} innerRadius={innerRadius} outerRadius={outerRadius + 8} startAngle={startAngle} endAngle={endAngle} fill={fill} />
        <Sector cx={cx} cy={cy} innerRadius={innerRadius - 4} outerRadius={innerRadius} startAngle={startAngle} endAngle={endAngle} fill={fill} />
      </g>
    );
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      return (
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', padding: '10px 14px', borderRadius: 10, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <p style={{ margin: 0, fontWeight: 700, color: '#1a202c' }}>{payload[0].name}</p>
          <p style={{ margin: 0, color: payload[0].payload.fill || '#4a5568' }}>Weight: <strong>{payload[0].value}%</strong></p>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
      <ResponsiveContainer width="100%" height={280}>
        <RechartsPie>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={70}
            outerRadius={110}
            dataKey="value"
            activeIndex={activeIndex}
            activeShape={renderActiveShape}
            onMouseEnter={(_, index) => setActiveIndex(index)}
            onMouseLeave={() => setActiveIndex(null)}
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />
            ))}
          </Pie>
          <RechartsTooltip content={<CustomTooltip />} />
          <Legend
            formatter={(value) => <span style={{ color: '#4a5568', fontSize: 12 }}>{value}</span>}
            iconType="circle"
            iconSize={10}
          />
        </RechartsPie>
      </ResponsiveContainer>
    </div>
  );
}

function RiskTrendChart({ rows }) {
  if (!rows.length) return <p className="muted">No risk history has been recorded yet.</p>;

  const data = averageDailyRiskRows(rows).map(row => ({
    date: new Date(row.created_at).toLocaleDateString(),
    score: Math.round(Number(row.risk_score || 0) * 100),
    level: row.risk_level,
    sampleCount: row.sample_count || 1,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: '#fff', border: '1px solid #cfd9e2', padding: '10px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <p style={{ margin: '0 0 5px 0', fontWeight: 'bold', color: '#344556' }}>{label}</p>
          <p style={{ margin: '0', color: '#b13d3d' }}>Risk Score: <strong>{payload[0].value}%</strong></p>
          <p style={{ margin: '0', color: '#657384', fontSize: '13px' }}>Level: {payload[0].payload.level}</p>
          <p style={{ margin: '0', color: '#657384', fontSize: '13px' }}>Daily average from {payload[0].payload.sampleCount} run(s)</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="trend-chart" style={{ height: '300px', width: '100%', marginTop: '20px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6edf2" vertical={false} />
          <XAxis dataKey="date" stroke="#657384" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis stroke="#657384" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(tick) => `${tick}%`} domain={[0, 100]} />
          <RechartsTooltip content={<CustomTooltip />} />
          <Line type="monotone" dataKey="score" stroke="#b13d3d" strokeWidth={3} dot={{ r: 5, fill: '#b13d3d', strokeWidth: 0 }} activeDot={{ r: 7 }} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function XaiPanel({ rows, decisionTrace }) {
  if (!rows.length && !decisionTrace.length) {
    return <p className="muted">XAI evidence will appear after modalities are processed.</p>;
  }
  return (
    <div className="xai-stack">
      {rows.map((row) => (
        <article key={`${row.modality}-${row.label}`} className="xai-row">
          <div>
            <strong>{row.label}</strong>
            <span>{row.modality}</span>
          </div>
          <p>{row.detail}</p>
        </article>
      ))}
      {decisionTrace.length > 0 && (
        <div className="xai-trace-summary">
          <strong>Decision Trace Context Passed to LLM</strong>
          <p>{decisionTrace.slice(0, 3).join(" | ")}</p>
        </div>
      )}
    </div>
  );
}

function buildXaiRows(latestRun, fusion) {
  const rows = [];
  const contributions = fusion?.modality_contributions || {};
  Object.entries(contributions)
    .filter(([, value]) => Number(value) > 0)
    .sort((left, right) => Number(right[1]) - Number(left[1]))
    .forEach(([modality, value]) => {
      rows.push({
        modality: prettyName(modality),
        label: "Fusion contribution",
        detail: `${formatPercent(value)} of the fused decision weight came from ${prettyName(modality)}.`,
      });
    });
  (latestRun?.module_outputs || []).forEach((output) => {
    const features = output.explanations?.top_features || [];
    features.slice(0, 3).forEach((feature) => {
      rows.push({
        modality: prettyName(output.modality),
        label: feature.feature || feature.name || "Model signal",
        detail: `${feature.direction || "Model effect"} with importance ${feature.importance_score ?? feature.shap_value ?? "n/a"}.`,
      });
    });
  });
  return rows.slice(0, 10);
}

function averageDailyRiskRows(rows) {
  const buckets = new Map();
  rows.forEach((row) => {
    const day = String(row.created_at || "").slice(0, 10);
    if (!day) return;
    const bucket = buckets.get(day) || { created_at: day, riskTotal: 0, confidenceTotal: 0, count: 0 };
    bucket.riskTotal += Number(row.risk_score || 0);
    bucket.confidenceTotal += Number(row.confidence || 0);
    bucket.count += 1;
    buckets.set(day, bucket);
  });
  return Array.from(buckets.values()).sort((a, b) => a.created_at.localeCompare(b.created_at)).map((bucket) => {
    const risk = bucket.riskTotal / Math.max(1, bucket.count);
    return {
      created_at: bucket.created_at,
      risk_score: risk,
      confidence: bucket.confidenceTotal / Math.max(1, bucket.count),
      risk_level: risk >= 0.7 ? "high" : risk >= 0.4 ? "medium" : "low",
      sample_count: bucket.count,
    };
  });
}

function DocumentPreview({ preview }) {
  if (!preview) {
    return <div className="document-preview muted">Select a document to preview it here.</div>;
  }
  if (preview.error) {
    return <div className="document-preview error">{preview.error}</div>;
  }
  const isImage = (preview.type || "").startsWith("image/") || /\.(png|jpe?g|webp|gif|tiff?)$/i.test(preview.filename || "");
  const isPdf = (preview.type || "").includes("pdf") || /\.pdf$/i.test(preview.filename || "");
  return (
    <div className="document-preview">
      <strong>{preview.filename}</strong>
      {isImage && <img src={preview.url} alt={preview.filename} />}
      {isPdf && <iframe title={preview.filename} src={preview.url} />}
      {!isImage && !isPdf && (
        <a className="secondary" href={preview.url} target="_blank" rel="noreferrer">
          Open file
        </a>
      )}
    </div>
  );
}

function buildReportDraft(data) {
  const fusion = data.latest_model_run?.fusion_output;
  if (!fusion) return "";
  const risk = fusion.risk || {};
  const diagnosis = fusion.diagnosis || {};
  const trace = (fusion.decision_trace || []).slice(0, 5).join("\n- ");
  return [
    `Patient ${data.patient.name || data.patient.patient_id} review draft`,
    `Patient ID: ${data.patient.patient_id}`,
    `Risk status: ${risk.class || "unknown"} (${formatPercent(risk.score)})`,
    `Diagnosis support: ${diagnosis.class || "unknown"} (${formatPercent(diagnosis.probability)})`,
    `Fusion confidence: ${formatPercent(fusion.confidence)}`,
    trace ? `Decision trace:\n- ${trace}` : "",
    "Clinician validation:",
  ].filter(Boolean).join("\n\n");
}

function prettyName(name) {
  return String(name || "").replace(/_/g, " ");
}

function guessMime(filename) {
  if (/\.pdf$/i.test(filename)) return "application/pdf";
  if (/\.(png|jpe?g|webp|gif|tiff?)$/i.test(filename)) return "image/*";
  return "application/octet-stream";
}

function formatPercent(val) {
  if (val === undefined || val === null || Number.isNaN(Number(val))) return "0%";
  return `${Math.round(Number(val) * 100)}%`;
}
