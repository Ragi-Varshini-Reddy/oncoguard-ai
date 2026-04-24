import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  Camera,
  ClipboardList,
  Download,
  FileUp,
  Microscope,
  ShieldCheck,
  Stethoscope,
  UserRound,
} from "lucide-react";
import {
  downloadPdf,
  getGenomicsSchema,
  getHealth,
  getRoles,
  inferFusion,
  inferGenomicsFile,
  trainGenomics,
} from "./api";
import "./styles.css";

const roleIcons = {
  patient: UserRound,
  doctor: Stethoscope,
  lab_technician: Microscope,
};

const phaseRoutes = [
  {
    id: "phase-1",
    hash: "#/phase-1",
    label: "Phase 1",
    title: "Intraoral + Clinical",
    description: "Image upload, clinical form, standalone outputs, and explanations.",
    icon: Camera,
  },
  {
    id: "phase-2",
    hash: "#/phase-2",
    label: "Phase 2",
    title: "Histopathology",
    description: "Patch/slide upload, pathology encoder, heatmaps, and contribution to fusion.",
    icon: Microscope,
  },
  {
    id: "phase-3-genomics",
    hash: "#/phase-3-genomics",
    label: "Phase 3",
    title: "Genomics",
    description: "Artifact-backed genomics training, inference, XAI, fusion, and reports.",
    icon: Brain,
  },
];

function routeFromHash() {
  const hash = window.location.hash || "#/phase-3-genomics";
  return phaseRoutes.some((route) => route.hash === hash) ? hash.slice(2) : "phase-3-genomics";
}

function App() {
  const [activeRoute, setActiveRoute] = useState(routeFromHash);
  const [role, setRole] = useState("doctor");
  const [roles, setRoles] = useState({});
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [patientId, setPatientId] = useState("");
  const [trainingFile, setTrainingFile] = useState(null);
  const [genomicFile, setGenomicFile] = useState(null);
  const [trainingSummary, setTrainingSummary] = useState(null);
  const [genomicsOutput, setGenomicsOutput] = useState(null);
  const [fusionOutput, setFusionOutput] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    function handleHashChange() {
      setActiveRoute(routeFromHash());
    }
    if (!window.location.hash) {
      window.history.replaceState(null, "", "#/phase-3-genomics");
    }
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    Promise.all([getHealth(), getRoles(), getGenomicsSchema()])
      .then(([healthData, roleData, schemaData]) => {
        setHealth(healthData);
        setRoles(roleData);
        setSchema(schemaData);
      })
      .catch((err) => setError(err.message));
  }, []);

  const currentRole = roles[role] || {};
  const canTrain = role === "lab_technician";
  const canInfer = role === "doctor" || role === "lab_technician";
  const canDownload = role === "doctor" || role === "patient";

  async function handleTrain() {
    if (!trainingFile) return;
    setBusy(true);
    setError("");
    try {
      const summary = await trainGenomics(trainingFile);
      setTrainingSummary(summary);
      setHealth(await getHealth());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleInfer() {
    if (!genomicFile) return;
    setBusy(true);
    setError("");
    setFusionOutput(null);
    try {
      const genomics = await inferGenomicsFile(genomicFile, patientId || undefined);
      setGenomicsOutput(genomics);
      if (genomics.status === "available") {
        const fusion = await inferFusion(genomics.patient_id, [genomics]);
        setFusionOutput(fusion);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDownload() {
    if (!genomicsOutput || !fusionOutput) return;
    await downloadPdf(genomicsOutput.patient_id, [genomicsOutput], fusionOutput);
  }

  const artifactReady = Boolean(health?.genomics_artifact_available);

  return (
    <main className="app">
      <section className="topbar">
        <div>
          <h1>OralCare-AI</h1>
          <p>Multimodal oral cancer decision-support workspace</p>
        </div>
        <div className={artifactReady ? "status ready" : "status missing"}>
          <ShieldCheck size={18} />
          {artifactReady ? "Genomics artifact ready" : "Train genomics model first"}
        </div>
      </section>

      <section className="disclaimer">
        <AlertTriangle size={18} />
        Decision support only. Outputs require clinician review and clinical validation before any real-world use.
      </section>

      {error && <section className="error">{error}</section>}

      <PhaseNav activeRoute={activeRoute} />

      {activeRoute === "phase-1" && (
        <PhasePlaceholder
          title="Phase 1 - Intraoral Imaging + Clinical Data"
          icon={Camera}
          description="This route is reserved for the teammate owning intraoral images and clinical structured data."
          bullets={[
            "Build image upload, preprocessing, pretrained vision encoder, and Grad-CAM.",
            "Build clinical form, tabular preprocessing, artifact-backed model, and feature explanations.",
            "Return contract-compatible intraoral and clinical module outputs for fusion.",
          ]}
          folder="Phase 1/README.md"
        />
      )}

      {activeRoute === "phase-2" && (
        <PhasePlaceholder
          title="Phase 2 - Histopathology"
          icon={Microscope}
          description="This route is reserved for the teammate owning histopathology patches/slides and pathology XAI."
          bullets={[
            "Build histopathology upload, patch preprocessing, and artifact-backed inference.",
            "Use pathology foundation embeddings or a fine-tuned vision model where feasible.",
            "Return contract-compatible histopathology outputs for fusion and reports.",
          ]}
          folder="Phase 2/README.md"
        />
      )}

      {activeRoute === "phase-3-genomics" && <section className="grid">
        <aside className="panel role-panel">
          <h2>Users</h2>
          <div className="role-list">
            {Object.entries(roles).map(([key, value]) => {
              const Icon = roleIcons[key] || UserRound;
              return (
                <button key={key} className={role === key ? "role selected" : "role"} onClick={() => setRole(key)}>
                  <Icon size={19} />
                  <span>{value.label}</span>
                </button>
              );
            })}
          </div>
          <h3>{currentRole.label || "Doctor"}</h3>
          <ul>
            {(currentRole.permissions || []).map((permission) => (
              <li key={permission}>{permission.replaceAll("_", " ")}</li>
            ))}
          </ul>
        </aside>

        <section className="panel workspace">
          <div className="panel-head">
            <div>
              <h2>Genomics Flow</h2>
              <p>CSV/parquet training artifacts, CSV inference, XAI, fusion, and report.</p>
            </div>
            <Brain size={28} />
          </div>

          <div className="two-col">
            <div className="subpanel">
              <h3>1. Train artifact</h3>
              <p className="muted">Training file must include patient ID, label, and the configured gene panel.</p>
              <FileInput disabled={!canTrain || busy} onChange={setTrainingFile} label="Training CSV" />
              <button className="primary" disabled={!canTrain || !trainingFile || busy} onClick={handleTrain}>
                <FileUp size={17} /> Train genomics model
              </button>
              {!canTrain && <p className="muted">Switch to Lab Technician to train or validate lab files.</p>}
            </div>

            <div className="subpanel">
              <h3>2. Run patient inference</h3>
              <input
                className="text-input"
                placeholder="Optional patient ID filter"
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
              />
              <FileInput disabled={!canInfer || busy} onChange={setGenomicFile} label="Patient genomic CSV" />
              <button className="primary" disabled={!canInfer || !genomicFile || busy} onClick={handleInfer}>
                <Activity size={17} /> Run genomics to fusion
              </button>
            </div>
          </div>

          <SchemaPanel schema={schema} />
          {trainingSummary && <TrainingSummary summary={trainingSummary} />}
          {genomicsOutput && <GenomicsResult output={genomicsOutput} />}
          {fusionOutput && <FusionResult output={fusionOutput} />}

          <div className="actions">
            <button className="secondary" disabled={!canDownload || !genomicsOutput || !fusionOutput} onClick={handleDownload}>
              <Download size={17} /> Download PDF report
            </button>
          </div>
        </section>
      </section>}
    </main>
  );
}

function PhaseNav({ activeRoute }) {
  return (
    <nav className="phase-nav" aria-label="Phase routes">
      {phaseRoutes.map((route) => {
        const Icon = route.icon;
        return (
          <a key={route.id} className={activeRoute === route.id ? "phase-link active" : "phase-link"} href={route.hash}>
            <Icon size={18} />
            <span>{route.label}</span>
            <strong>{route.title}</strong>
          </a>
        );
      })}
    </nav>
  );
}

function PhasePlaceholder({ title, icon: Icon, description, bullets, folder }) {
  return (
    <section className="phase-placeholder panel">
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        <Icon size={30} />
      </div>
      <div className="placeholder-grid">
        <div className="subpanel">
          <h3>Team Ownership</h3>
          <ul>
            {bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
        </div>
        <div className="subpanel">
          <h3>Where To Start</h3>
          <p className="muted">Read the phase folder before coding so the API contracts and responsibilities stay aligned.</p>
          <code>{folder}</code>
        </div>
      </div>
    </section>
  );
}

function FileInput({ label, onChange, disabled }) {
  return (
    <label className={disabled ? "file disabled" : "file"}>
      <FileUp size={18} />
      <span>{label}</span>
      <input disabled={disabled} type="file" accept=".csv,.parquet" onChange={(event) => onChange(event.target.files?.[0] || null)} />
    </label>
  );
}

function SchemaPanel({ schema }) {
  if (!schema) return null;
  return (
    <div className="schema">
      <ClipboardList size={18} />
      <div>
        <strong>Required genomics columns</strong>
        <p>{schema.required_columns_for_inference.join(", ")}</p>
      </div>
    </div>
  );
}

function TrainingSummary({ summary }) {
  const metrics = summary.metrics || {};
  return (
    <section className="result-block">
      <h3>Training Summary</h3>
      <div className="metrics">
        <Metric label="Rows" value={metrics.rows} />
        <Metric label="Accuracy" value={formatMetric(metrics.accuracy)} />
        <Metric label="F1" value={formatMetric(metrics.f1)} />
        <Metric label="ROC-AUC" value={formatMetric(metrics.roc_auc)} />
      </div>
      <p className="muted">{summary.model_card?.intended_use}</p>
    </section>
  );
}

function GenomicsResult({ output }) {
  const prediction = output.prediction || {};
  return (
    <section className="result-block">
      <h3>Genomics Output</h3>
      <div className="metrics">
        <Metric label="Status" value={output.status} />
        <Metric label="Risk" value={prediction.risk_class || "unavailable"} />
        <Metric label="Risk score" value={formatPercent(prediction.risk_score)} />
        <Metric label="Confidence" value={formatPercent(output.confidence)} />
      </div>
      {output.status === "error" && <p className="error inline">{output.warnings?.join(" ")}</p>}
      {output.explanations?.top_features?.length > 0 && <FeatureTable rows={output.explanations.top_features} />}
    </section>
  );
}

function FusionResult({ output }) {
  const contributions = output.modality_contributions || {};
  const diagnosis = output.diagnosis || {};
  const risk = output.risk || {};
  return (
    <section className="result-block">
      <h3>Fusion + Report Output</h3>
      <div className="metrics">
        <Metric label="Diagnosis" value={diagnosis.class} />
        <Metric label="Probability" value={formatPercent(diagnosis.probability)} />
        <Metric label="Risk" value={risk.class} />
        <Metric label="Confidence" value={formatPercent(output.confidence)} />
      </div>
      <div className="bars">
        {Object.entries(contributions).map(([name, value]) => (
          <div key={name} className="bar-row">
            <span>{name}</span>
            <div className="bar-track"><div style={{ width: `${Math.round(value * 100)}%` }} /></div>
            <strong>{formatPercent(value)}</strong>
          </div>
        ))}
      </div>
      {output.warnings?.length > 0 && (
        <ul className="warnings">
          {output.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </section>
  );
}

function FeatureTable({ rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>Value</th>
            <th>Importance</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.feature}>
              <td>{row.feature}</td>
              <td>{row.value ?? "imputed"}</td>
              <td>{row.importance_score}</td>
              <td>{row.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "n/a"}</strong>
    </div>
  );
}

function formatPercent(value) {
  if (typeof value !== "number") return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatMetric(value) {
  if (typeof value !== "number") return "n/a";
  return String(value);
}

createRoot(document.getElementById("root")).render(<App />);
