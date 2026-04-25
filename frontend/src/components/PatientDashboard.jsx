import React, { useEffect, useMemo, useState } from "react";
import {
  Heart,
  MessageSquare,
  FileSearch,
  Stethoscope,
  Microscope,
  Info,
  RefreshCw,
  Download,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Upload,
  Camera,
  CheckCircle2,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart as RechartsPie,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { downloadApprovedReport, getPatient, getPatientRiskHistory, uploadPatientDocument } from "../services/api";
import AppointmentsPanel from "./AppointmentsPanel";
import PatientChat from "./PatientChat";

const CHART_COLORS = ["#236b75", "#b13d3d", "#d99a00", "#5b6bf1", "#1b7046", "#7e57c2"];

function formatPercent(value, digits = 0) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "—";
  const scaled = numeric <= 1 ? numeric * 100 : numeric;
  return `${scaled.toFixed(digits)}%`;
}

function getClinicalOutput(latestRun) {
  return latestRun?.module_outputs?.find((output) => output.modality === "clinical" && output.status === "available") || null;
}

function getFeatureImportance(topFeatures, featureName) {
  const item = topFeatures.find((entry) => entry.feature === featureName);
  const rawValue = item?.importance_score ?? item?.shap_value ?? 0;
  const numeric = Number(rawValue);
  return Number.isNaN(numeric) ? 0 : Math.abs(numeric);
}

function estimateLowerRisk(currentRiskPercent, featureImportance, totalImportance) {
  if (currentRiskPercent == null) return null;
  if (featureImportance <= 0) return currentRiskPercent;
  const denominator = totalImportance > 0 ? totalImportance : featureImportance;
  const share = denominator > 0 ? featureImportance / denominator : 0;
  const reductionFraction = Math.min(0.25, Math.max(0.08, share * 0.35));
  return Math.max(0, Math.round(currentRiskPercent * (1 - reductionFraction)));
}

function buildSuggestionCards(latestRun) {
  const fusion = latestRun?.fusion_output || null;
  const clinical = getClinicalOutput(latestRun);
  const currentRiskPercent = fusion?.risk?.score != null ? Math.round(Number(fusion.risk.score) * 100) : null;
  const cards = [];

  if (clinical) {
    const featureValues = clinical.explanations?.feature_values || {};
    const topFeatures = clinical.explanations?.top_features || [];
    const habitCandidates = ["tobacco_use", "alcohol_use"].filter((feature) => featureValues[feature]);
    if (habitCandidates.length && currentRiskPercent != null) {
      const habitFeature = habitCandidates.length === 2
        ? habitCandidates.sort((left, right) => getFeatureImportance(topFeatures, right) - getFeatureImportance(topFeatures, left))[0]
        : habitCandidates[0];
      const habitImportance = getFeatureImportance(topFeatures, habitFeature);
      const totalImportance = topFeatures.reduce((sum, entry) => sum + Math.max(0, Number(entry.importance_score ?? entry.shap_value ?? 0) || 0), 0);
      const adjustedRisk = estimateLowerRisk(currentRiskPercent, habitImportance, totalImportance);
      const featureLabel = habitFeature === "tobacco_use" ? "smoking" : "alcohol use";
      cards.push({
        tone: "bad-habit",
        title: habitFeature === "tobacco_use" ? "Quit smoking" : "Reduce or stop alcohol",
        metric: `${formatPercent(currentRiskPercent)} current risk • SHAP estimate after change: ${adjustedRisk != null ? formatPercent(adjustedRisk) : "—"}`,
        detail: `${featureLabel} is one of the strongest positive SHAP drivers in the latest clinical explanation.`,
      });
    }
  }

  cards.push({
    tone: "new-habit",
    title: "Build a daily oral self-check",
    metric: "Target: 7 days/week",
    detail: clinical?.explanations?.feature_values?.poor_oral_hygiene
      ? "Pair brushing and flossing with a 2-minute mouth check so bleeding, sores, or white/red patches are noticed earlier."
      : "Add a 2-minute oral self-check each day so new changes are noticed sooner and can be raised with your clinician.",
  });

  return cards;
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

function buildPatientXaiRows(latestRun, fusion) {
  const rows = [];
  const contributions = fusion?.modality_contributions || {};
  Object.entries(contributions)
    .filter(([, value]) => Number(value) > 0)
    .sort((left, right) => Number(right[1]) - Number(left[1]))
    .slice(0, 4)
    .forEach(([modality, value]) => {
      rows.push({
        modality: modality.replace(/_/g, " "),
        label: "Main evidence source",
        detail: `${formatPercent(value)} of the AI's combined score came from this report type.`,
      });
    });
  (latestRun?.module_outputs || []).forEach((output) => {
    const features = output.explanations?.top_features || [];
    features.slice(0, 2).forEach((feature) => {
      rows.push({
        modality: output.modality.replace(/_/g, " "),
        label: feature.feature || feature.name || "Model signal",
        detail: "This was one of the signals the AI used. Please ask your doctor what it means for you.",
      });
    });
  });
  return rows.slice(0, 8);
}

function groupIntraoralUploadsByDate(documents) {
  const buckets = new Map();
  documents
    .filter((doc) => doc.document_type === "intraoral")
    .forEach((doc) => {
      const date = String(doc.created_at || "").slice(0, 10);
      if (!date) return;
      const bucket = buckets.get(date) || [];
      bucket.push(doc);
      buckets.set(date, bucket);
    });
  return Array.from(buckets.entries())
    .sort((left, right) => right[0].localeCompare(left[0]))
    .map(([date, items]) => ({ date, items }));
}

export default function PatientDashboard({ user, view = "dashboard" }) {
  const [data, setData] = useState(null);
  const [riskHistory, setRiskHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showChat, setShowChat] = useState(false);
  const [dailyFile, setDailyFile] = useState(null);
  const [dailyNotes, setDailyNotes] = useState("");
  const [dailyBusy, setDailyBusy] = useState(false);
  const [dailyStatus, setDailyStatus] = useState("");
  const [dailyInsight, setDailyInsight] = useState(null);
  const patientId = user?.profile?.patient_id;

  useEffect(() => {
    loadData();
  }, [patientId]);

  async function loadData() {
    if (!patientId) {
      setError("Patient profile is not linked to this demo login.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const [patientData, historyData] = await Promise.all([
        getPatient(patientId),
        getPatientRiskHistory(patientId).catch((err) => {
          console.warn("Risk history unavailable", err);
          return { risk_history: [] };
        }),
      ]);
      setData(patientData);
      setRiskHistory(historyData?.risk_history || []);
    } catch (err) {
      console.error("Failed to load patient data", err);
      setError(err.message || "Could not load patient record.");
      setData(null);
      setRiskHistory([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleDailyUpload(event) {
    event.preventDefault();
    if (!dailyFile || !patientId || dailyBusy) return;
    setDailyBusy(true);
    setDailyStatus("Analyzing your intraoral image...");
    setDailyInsight(null);
    try {
      const today = new Date().toISOString().slice(0, 10);
      const response = await uploadPatientDocument(
        patientId,
        dailyFile,
        "intraoral",
        `Patient daily intraoral upload ${today}${dailyNotes ? `: ${dailyNotes}` : ""}`,
      );
      setDailyInsight(response);
      setDailyStatus(response.processing_warning || "Image analyzed. Your daily risk trend has been updated.");
      setDailyFile(null);
      setDailyNotes("");
      await loadData();
    } catch (err) {
      setDailyStatus(err.message || "Could not analyze this image.");
    } finally {
      setDailyBusy(false);
    }
  }

  const latestRun = data?.latest_model_run;
  const fusion = latestRun?.fusion_output || null;
  const suggestionCards = useMemo(() => buildSuggestionCards(latestRun), [latestRun]);
  const chartHistory = useMemo(() => {
    const sourceRows = riskHistory.length > 0
      ? riskHistory
      : latestRun
        ? [{ created_at: latestRun.created_at, risk_score: fusion?.risk?.score, risk_level: fusion?.risk?.class || "reviewed" }]
        : [];
    return averageDailyRiskRows(sourceRows)
      .filter((row) => row && row.created_at != null)
      .map((row) => ({
        date: new Date(row.created_at).toLocaleDateString(),
        score: Math.round(Number(row.risk_score || 0) * 100),
        label: row.risk_level || "reviewed",
        sampleCount: row.sample_count || 1,
      }));
  }, [riskHistory, latestRun, fusion]);
  const contributionChart = useMemo(() => {
    const contributions = fusion?.modality_contributions || {};
    return Object.entries(contributions)
      .filter(([, value]) => Number(value) > 0)
      .sort((left, right) => Number(right[1]) - Number(left[1]))
      .map(([name, value]) => ({ name, value: Math.round(Number(value) * 100) }));
  }, [fusion]);
  const riskHeads = useMemo(() => {
    const diagnosis = fusion?.diagnosis || {};
    const risk = fusion?.risk || {};
    return [
      {
        label: "Diagnosis Head",
        value: (diagnosis.class || "reviewed").toUpperCase(),
        accent: "blue",
        detail: `Support ${formatPercent(diagnosis.probability)}`,
      },
      {
        label: "Risk Head",
        value: (risk.class || "reviewed").toUpperCase(),
        accent: "red",
        detail: `Current risk ${formatPercent(risk.score)}`,
      },
      {
        label: "Confidence Head",
        value: formatPercent(fusion?.confidence),
        accent: "blue",
        detail: "Fusion confidence",
      },
    ];
  }, [fusion]);
  const xaiRows = useMemo(() => buildPatientXaiRows(latestRun, fusion), [latestRun, fusion]);
  const dailyUploads = useMemo(() => groupIntraoralUploadsByDate(data?.documents || []), [data]);

  if (loading) return <div className="loader">Loading your health records...</div>;

  if (error || !data) {
    return (
      <div className="dashboard patient-dashboard">
        <header className="dashboard-header">
          <div className="header-info">
            <h1>{view === "reports" ? "My Reports" : "My Health Portal"}</h1>
            <p>{view === "reports" ? "Approved reports, status, and PDF downloads" : "Secure access to your diagnostic journey"}</p>
          </div>
        </header>
        <section className="empty-state panel">
          <Info size={42} />
          <h3>Patient record unavailable</h3>
          <p>{error || "This demo login is not linked to an accessible patient record."}</p>
          <button className="secondary" onClick={loadData}>
            <RefreshCw size={17} /> Retry
          </button>
        </section>
      </div>
    );
  }

  return (
    <div className="dashboard patient-dashboard">
      <header className="dashboard-header">
        <div className="header-info">
          <h1>{view === "reports" ? "My Reports" : "My Health Portal"}</h1>
          <p>{view === "reports" ? "Approved reports, status, and PDF downloads" : "Secure access to your diagnostic journey"}</p>
        </div>
        <div className="team-card">
          <div className="team-member">
            <Stethoscope size={16} />
            <span>Dr. {data.doctor?.name || "Anika Rao"}</span>
          </div>
          <div className="team-member">
            <Microscope size={16} />
            <span>Tech: {data.technician?.name || "Ravi"}</span>
          </div>
        </div>
      </header>

      <div className="patient-layout-grid">
        <main className="health-timeline">
          <section className="welcome-banner panel">
            <Heart size={32} className="heart-icon" />
            <div className="banner-text">
              <h2>Welcome back, {data.patient.name || data.patient.patient_id}</h2>
              <p>Your latest diagnostic updates are available for review.</p>
            </div>
          </section>

          <section className="panel daily-upload-panel">
            <div className="panel-head">
              <Camera size={20} />
              <h3>Daily Intraoral AI Check</h3>
            </div>
            <form className="daily-upload-form" onSubmit={handleDailyUpload}>
              <label className="daily-file-picker">
                <input
                  type="file"
                  accept="image/*,.tif,.tiff"
                  onChange={(event) => setDailyFile(event.target.files?.[0] || null)}
                />
                <Upload size={22} />
                <span>{dailyFile ? dailyFile.name : "Upload today’s intraoral image"}</span>
              </label>
              <input
                value={dailyNotes}
                onChange={(event) => setDailyNotes(event.target.value)}
                placeholder="Optional note, symptom, or lesion location"
              />
              <button className="primary" disabled={!dailyFile || dailyBusy}>
                {dailyBusy ? "Analyzing..." : "Submit"}
              </button>
            </form>
            {dailyStatus && (
              <div className={`status-alert ${dailyInsight?.processing_warning ? "error" : "success"}`}>
                {dailyInsight?.processing_warning ? <Info size={18} /> : <CheckCircle2 size={18} />}
                <span>{dailyStatus}</span>
              </div>
            )}
            {dailyInsight?.fusion_output && (
              <div className="daily-insight-grid">
                <div className="highlight">
                  <span>Daily Risk</span>
                  <strong className={dailyInsight.fusion_output.risk?.class}>
                    {dailyInsight.fusion_output.risk?.class?.toUpperCase()}
                  </strong>
                </div>
                <div className="highlight">
                  <span>Risk Score</span>
                  <strong>{formatPercent(dailyInsight.fusion_output.risk?.score)}</strong>
                </div>
                <div className="highlight">
                  <span>AI Confidence</span>
                  <strong>{formatPercent(dailyInsight.fusion_output.confidence)}</strong>
                </div>
              </div>
            )}
            <div className="daily-upload-history">
              <strong>Date-wise intraoral uploads</strong>
              {dailyUploads.length ? dailyUploads.map((group) => (
                <div key={group.date} className="daily-upload-day">
                  <span>{new Date(group.date).toLocaleDateString()}</span>
                  <small>{group.items.length} image{group.items.length === 1 ? "" : "s"} stored</small>
                </div>
              )) : <p className="muted">No daily intraoral uploads yet.</p>}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <ShieldCheck size={20} />
              <h3>Risk Heads</h3>
            </div>
            <div className="quick-stats patient-head-grid">
              {riskHeads.map((item) => (
                <div key={item.label} className={`stat-card ${item.accent}`}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                  <small className="muted">{item.detail}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <TrendingUp size={20} />
              <h3>Charts</h3>
            </div>
            <div className="patient-chart-grid">
              <div className="patient-chart-card">
                <div className="trend-labels">
                  <span>Risk Trend</span>
                  <span>{chartHistory.length ? `${chartHistory.length} points` : "No history yet"}</span>
                </div>
                {chartHistory.length ? (
                  <div className="trend-chart">
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={chartHistory} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e6edf2" vertical={false} />
                        <XAxis dataKey="date" stroke="#657384" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="#657384" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(tick) => `${tick}%`} domain={[0, 100]} />
                        <RechartsTooltip
                          formatter={(value) => `${value}%`}
                          labelFormatter={(label) => `Daily average: ${label}`}
                        />
                        <Line type="monotone" dataKey="score" stroke="#b13d3d" strokeWidth={3} dot={{ r: 4, fill: "#b13d3d", strokeWidth: 0 }} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="muted">No risk history has been recorded yet.</p>
                )}
              </div>

              <div className="patient-chart-card">
                <div className="trend-labels">
                  <span>Modality Contribution</span>
                  <span>{contributionChart.length ? `${contributionChart.length} modalities` : "No fused output yet"}</span>
                </div>
                {contributionChart.length ? (
                  <div className="trend-chart">
                    <ResponsiveContainer width="100%" height={260}>
                      <RechartsPie>
                        <Pie data={contributionChart} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={92} innerRadius={56} stroke="none">
                          {contributionChart.map((entry, index) => (
                            <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                        <RechartsTooltip formatter={(value, name) => [`${value}%`, name]} />
                        <Legend />
                      </RechartsPie>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="muted">No modality contribution chart is available yet.</p>
                )}
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <Sparkles size={20} />
              <h3>Actionable Suggestions</h3>
            </div>
            <div className="suggestion-stack">
              {suggestionCards.map((item) => (
                <article key={item.title} className={`suggestion-card ${item.tone}`}>
                  <div className="suggestion-card-head">
                    <strong>{item.title}</strong>
                    <span className="suggestion-metric">{item.metric}</span>
                  </div>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <ShieldCheck size={20} />
              <h3>Why the AI Flagged This</h3>
            </div>
            <div className="xai-stack patient-xai">
              {xaiRows.map((row) => (
                <article key={`${row.modality}-${row.label}`} className="xai-row">
                  <div>
                    <strong>{row.label}</strong>
                    <span>{row.modality}</span>
                  </div>
                  <p>{row.detail}</p>
                </article>
              ))}
              {xaiRows.length === 0 && <p className="muted">The explanation will appear after your reports are processed.</p>}
            </div>
          </section>

          <div className="report-summary panel">
            <div className="panel-head">
              <FileSearch size={20} />
              <h3>Your Latest Results</h3>
            </div>
            {data.latest_approval?.approval_status === "approved" ? (
              <div className="result-highlights">
                <div className="highlight">
                  <span>Doctor Status</span>
                  <strong className="low">APPROVED</strong>
                </div>
                <div className="highlight">
                  <span>Risk Status</span>
                  <strong className={data.latest_model_run?.fusion_output?.risk?.class}>
                    {data.latest_model_run?.fusion_output?.risk?.class?.toUpperCase() || "REVIEWED"}
                  </strong>
                </div>
                <p className="approved-report">
                  {data.latest_approval.report_text}
                </p>
                {data.latest_approval.doctor_notes && (
                  <p className="doctor-note">Doctor note: {data.latest_approval.doctor_notes}</p>
                )}
                <button className="primary report-download" onClick={() => downloadApprovedReport(data.patient.patient_id)}>
                  <Download size={17} /> Download Approved PDF
                </button>
              </div>
            ) : data.latest_model_run ? (
              <div className="result-highlights">
                <div className="highlight">
                  <span>Review Status</span>
                  <strong className="medium">
                    {(data.latest_approval?.approval_status || "PENDING").toUpperCase()}
                  </strong>
                </div>
                <div className="highlight">
                  <span>Confidence</span>
                  <strong>{formatPercent(data.latest_model_run.fusion_output.confidence)}</strong>
                </div>
                <p className="disclaimer">
                  <Info size={14} />
                  Your results are being reviewed by Dr. {data.doctor?.name}. The approved summary will appear here after validation.
                </p>
              </div>
            ) : (
              <p className="muted">Your reports are currently being processed by the lab.</p>
            )}
          </div>

          <AppointmentsPanel user={user} patientId={data.patient.patient_id} />
        </main>

        <aside className="assistant-sidebar">
          {!showChat ? (
            <div className="chat-teaser panel">
              <MessageSquare size={32} />
              <h3>AI Health Assistant</h3>
              <p>Have questions about your reports? Our AI can help explain technical terms and findings.</p>
              <button className="primary" onClick={() => setShowChat(true)}>
                Start Consultation
              </button>
            </div>
          ) : (
            <div className="chat-container panel">
              <div className="panel-head">
                <MessageSquare size={20} />
                <h3>AI Consultation</h3>
                <button className="close-btn" onClick={() => setShowChat(false)}>×</button>
              </div>
              <PatientChat
                patientId={data.patient.patient_id}
                latestRun={data.latest_model_run}
              />
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
