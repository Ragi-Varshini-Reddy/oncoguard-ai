import React, { useEffect, useRef, useState } from "react";
import { 
  Users, 
  Upload, 
  FilePlus, 
  Camera, 
  Microscope, 
  Brain, 
  CheckCircle2, 
  AlertCircle,
  ClipboardList,
  ArrowLeft
} from "lucide-react";
import { getTechnicianPatients, submitClinicalData, uploadPatientDocument, recordProcessingAppointment } from "../services/api";

export default function TechnicianDashboard() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [uploadType, setUploadType] = useState("intraoral");
  
  // Track selected files and notes for each modality
  const [uploads, setUploads] = useState({
    intraoral: { file: null, notes: "" },
    histopathological: { file: null, notes: "" },
    genomic: { file: null, notes: "" },
    clinical: { active: false, notes: "" }
  });

  const [status, setStatus] = useState({ type: "", message: "" });
  const [latestResult, setLatestResult] = useState(null);
  const [clinical, setClinical] = useState({
    tobacco_use: false,
    alcohol_use: false,
    lesion_site: "lateral tongue",
    lesion_size_cm: "",
    persistent_ulcer_weeks: "",
    neck_node_present: false,
    poor_oral_hygiene: false,
    family_history: false,
  });
  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadPatients();
  }, []);

  async function loadPatients() {
    try {
      const data = await getTechnicianPatients();
      setPatients(data.patients);
    } catch (err) {
      console.error("Failed to load patients", err);
    }
  }

  function handleFileChange(e) {
    if (e.target.files && e.target.files[0]) {
      setUploads(prev => ({
        ...prev,
        [uploadType]: { ...prev[uploadType], file: e.target.files[0] }
      }));
    }
  }

  function handleNotesChange(e) {
    setUploads(prev => ({
      ...prev,
      [uploadType]: { ...prev[uploadType], notes: e.target.value }
    }));
  }

  async function handleProcess(e) {
    e.preventDefault();
    if (!selectedPatient) return;

    const toProcess = Object.entries(uploads).filter(([key, val]) => {
      if (key === "clinical") return val.active;
      return val.file !== null;
    });

    if (toProcess.length === 0) {
      setStatus({ type: "error", message: "Please add at least one modality to process." });
      return;
    }

    setBusy(true);
    setLatestResult(null);
    setStatus({ type: "info", message: `Processing ${toProcess.length} modalities...` });
    
    let lastResponse = null;
    try {
      for (const [type, data] of toProcess) {
        if (type === "clinical") {
          lastResponse = await submitClinicalData(selectedPatient.patient_id, normalizedClinical());
        } else {
          lastResponse = await uploadPatientDocument(selectedPatient.patient_id, data.file, type, data.notes);
        }
      }
      
      // Track appointment
      await recordProcessingAppointment(selectedPatient.patient_id);
      
      setLatestResult(lastResponse);
      setStatus({
        type: lastResponse.processing_warning ? "error" : "success",
        message: lastResponse.processing_warning || `Successfully processed ${toProcess.length} modalities.`,
      });
      
      // Reset uploads
      setUploads({
        intraoral: { file: null, notes: "" },
        histopathological: { file: null, notes: "" },
        genomic: { file: null, notes: "" },
        clinical: { active: false, notes: "" }
      });
    } catch (err) {
      setStatus({ type: "error", message: err.message });
    } finally {
      setBusy(false);
    }
  }

  function chooseModality(type) {
    setUploadType(type);
    setStatus({ type: "", message: "" });
    setLatestResult(null);
    
    if (type === "clinical") {
      setUploads(prev => ({
        ...prev,
        clinical: { ...prev.clinical, active: true }
      }));
    } else if (!uploads[type].file) {
      window.setTimeout(() => fileInputRef.current?.click(), 0);
    }
  }

  function normalizedClinical() {
    return {
      ...clinical,
      lesion_size_cm: clinical.lesion_size_cm === "" ? undefined : Number(clinical.lesion_size_cm),
      persistent_ulcer_weeks: clinical.persistent_ulcer_weeks === "" ? undefined : Number(clinical.persistent_ulcer_weeks),
    };
  }

  function updateClinical(field, value) {
    setClinical(prev => ({ ...prev, [field]: value }));
  }
  
  function goBack() {
    setSelectedPatient(null);
    setStatus({ type: "", message: "" });
    setLatestResult(null);
  }

  const hasFilesToProcess = Object.values(uploads).some(u => u.file !== null) || uploads.clinical.active;

  return (
    <div className="dashboard technician-dashboard">
      <header className="dashboard-header">
        <div className="header-info">
          <h1>Technician Workspace</h1>
          <p>Manage patient records and batch diagnostic uploads</p>
        </div>
      </header>

      <div className={`dashboard-grid ${selectedPatient ? "patient-selected" : ""}`}>
        {!selectedPatient && (
          <aside className="patient-sidebar panel">
            <div className="panel-head">
              <Users size={20} />
              <h2>Assigned Patients</h2>
            </div>
            <div className="patient-list">
              {patients.map(p => (
                <button 
                  key={p.patient_id} 
                  className={selectedPatient?.patient_id === p.patient_id ? "patient-card active" : "patient-card"}
                  onClick={() => setSelectedPatient(p)}
                >
                  <strong>{p.name || p.patient_id}</strong>
                  <span>{p.patient_id} · {p.sex}, {p.age}y</span>
                </button>
              ))}
            </div>
          </aside>
        )}

        <main className="upload-workspace panel" style={{ gridColumn: selectedPatient ? "1 / -1" : "auto" }}>
          {!selectedPatient ? (
            <div className="empty-state">
              <Users size={48} />
              <h3>Select a patient to begin</h3>
              <p>Choose a patient from the left sidebar to upload diagnostic data.</p>
            </div>
          ) : (
            <div className="upload-container">
              <div className="panel-head">
                <button className="back-btn" onClick={goBack} title="Back to Patients">
                  <ArrowLeft size={20} />
                </button>
                <FilePlus size={20} style={{ marginLeft: "12px" }}/>
                <h2>Batch Upload: {selectedPatient.name || selectedPatient.patient_id}</h2>
              </div>

              <form className="upload-form" onSubmit={handleProcess}>
                <div className="form-group">
                  <label>Select Modality to Configure</label>
                  <div className="type-selector">
                    {[
                      { id: "intraoral", label: "Intraoral Image", icon: Camera },
                      { id: "histopathological", label: "Histopathology", icon: Microscope },
                      { id: "genomic", label: "Genomic CSV", icon: Brain },
                      { id: "clinical", label: "Clinical Data", icon: ClipboardList },
                    ].map(t => {
                      const isActiveModality = uploadType === t.id;
                      const hasData = t.id === "clinical" ? uploads.clinical.active : uploads[t.id].file !== null;
                      return (
                        <button 
                          key={t.id}
                          type="button"
                          className={`type-btn ${isActiveModality ? "active" : ""} ${hasData ? "has-data" : ""}`}
                          onClick={() => chooseModality(t.id)}
                        >
                          <t.icon size={18} />
                          <span>{t.label}</span>
                          {hasData && <CheckCircle2 size={14} className="data-check" />}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {uploadType !== "clinical" ? (
                  <div className="form-group slide-in">
                    <label>Selected File for {uploadType}</label>
                    <div className="file-dropzone" onClick={() => fileInputRef.current?.click()}>
                      <input
                        ref={fileInputRef}
                        type="file"
                        onChange={handleFileChange}
                        className="hidden"
                        accept={uploadType === "genomic" ? ".csv" : "image/*,.tif,.tiff"}
                      />
                      <div className="dropzone-label">
                        <Upload size={24} />
                        {uploads[uploadType].file ? (
                          <strong>{uploads[uploadType].file.name}</strong>
                        ) : (
                          <span>Click to browse for {uploadType} file</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="clinical-form slide-in">
                    <div className="form-group" style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                      <strong>Configure Clinical Data</strong>
                      <button type="button" className="btn-small" onClick={() => setUploads(p => ({...p, clinical: {...p.clinical, active: false}}))}>
                        Clear
                      </button>
                    </div>
                    <div className="status-alert info" style={{ marginBottom: "0.75rem" }}>
                      <span>
                        Using patient demographics from the selected record: {selectedPatient.sex}, {selectedPatient.age}y.
                      </span>
                    </div>
                    <label>Lesion Site<input value={clinical.lesion_site} onChange={e => updateClinical("lesion_site", e.target.value)} /></label>
                    <label>Lesion Size (cm)<input value={clinical.lesion_size_cm} onChange={e => updateClinical("lesion_size_cm", e.target.value)} type="number" step="0.1" min="0" /></label>
                    <label>Ulcer Weeks<input value={clinical.persistent_ulcer_weeks} onChange={e => updateClinical("persistent_ulcer_weeks", e.target.value)} type="number" step="1" min="0" /></label>
                    {[
                      ["tobacco_use", "Tobacco use"],
                      ["alcohol_use", "Alcohol use"],
                      ["neck_node_present", "Neck node present"],
                      ["poor_oral_hygiene", "Poor oral hygiene"],
                      ["family_history", "Family history"],
                    ].map(([field, label]) => (
                      <label key={field} className="check-row">
                        <input type="checkbox" checked={clinical[field]} onChange={e => updateClinical(field, e.target.checked)} />
                        <span>{label}</span>
                      </label>
                    ))}
                  </div>
                )}

                <div className="form-group">
                  <label>Notes for {uploadType}</label>
                  <textarea 
                    value={uploads[uploadType].notes} 
                    onChange={handleNotesChange}
                    placeholder="Enter clinical observations or processing notes..."
                  />
                </div>

                {status.message && (
                  <div className={`status-alert ${status.type}`}>
                    {status.type === "success" ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
                    <span>{status.message}</span>
                  </div>
                )}

                {busy && <LabProcessingAnimation />}

                {latestResult?.fusion_output && (
                  <div className="processing-result">
                    <strong>Batch Processing Complete</strong>
                    <span>Risk: {latestResult.fusion_output.risk?.class} ({Math.round((latestResult.fusion_output.risk?.score || 0) * 100)}%)</span>
                    <span>Confidence: {Math.round((latestResult.fusion_output.confidence || 0) * 100)}%</span>
                  </div>
                )}

                <button className="primary submit-btn" disabled={busy || !hasFilesToProcess}>
                  {busy ? "Processing Batch..." : "Process All Configured Modalities"}
                </button>
              </form>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function LabProcessingAnimation() {
  return (
    <div className="lab-processing-scene css-animated">
      <div className="lab-animation-wrapper">
        <svg viewBox="0 0 200 100" width="100%" height="150" className="lab-svg">
          {/* Flask 1 */}
          <path d="M40 80 L60 80 L55 50 L55 30 L45 30 L45 50 Z" fill="none" stroke="#5a8f7b" strokeWidth="2" />
          <path className="liquid-1" d="M42 78 L58 78 L54 55 L46 55 Z" fill="#5a8f7b" opacity="0.6" />
          {/* Bubbles in Flask 1 */}
          <circle className="bubble bubble-1" cx="50" cy="70" r="2" fill="#fff" />
          <circle className="bubble bubble-2" cx="48" cy="65" r="1.5" fill="#fff" />
          
          {/* Flask 2 (Microscope approximation) */}
          <path d="M110 80 L140 80 L130 70 L130 50 L115 30 L110 35 L120 55 L120 70 Z" fill="none" stroke="#236b75" strokeWidth="2" />
          <circle cx="115" cy="40" r="5" fill="none" stroke="#b13d3d" strokeWidth="2" />
          <line x1="125" y1="75" x2="135" y2="75" stroke="#236b75" strokeWidth="2" />
          
          {/* Connecting data lines */}
          <path className="data-line" d="M65 55 Q 85 30 105 45" fill="none" stroke="#79aeb6" strokeWidth="1" strokeDasharray="4 4" />
          <path className="data-line delay" d="M65 65 Q 85 85 105 65" fill="none" stroke="#79aeb6" strokeWidth="1" strokeDasharray="4 4" />
          
          {/* Scanner beam */}
          <line className="scanner-beam" x1="100" y1="20" x2="150" y2="20" stroke="#b13d3d" strokeWidth="1" opacity="0.5" />
        </svg>
      </div>
      <div style={{ textAlign: "center", marginTop: "1rem" }}>
        <strong>Processing multimodal evidence</strong>
        <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-light)" }}>Running batch analysis, updating fusion, and saving context.</p>
      </div>
    </div>
  );
}
