import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Brain,
  CalendarClock,
  ClipboardList,
  Microscope,
  ShieldCheck,
  Stethoscope,
  UserRound,
} from "lucide-react";
import {
  demoLogin,
  getDemoUsers,
} from "./services/api";
import "./assets/styles.css";

const roleIcons = {
  patient: UserRound,
  doctor: Stethoscope,
  lab_technician: Microscope,
};

import TechnicianDashboard from "./components/TechnicianDashboard";
import DoctorDashboard from "./components/DoctorDashboard";
import PatientDashboard from "./components/PatientDashboard";
import AppointmentsPanel from "./components/AppointmentsPanel";

function App() {
  const [user, setUser] = useState(null);
  const [demoUsers, setDemoUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("dashboard");

  useEffect(() => {
    const stored = localStorage.getItem("oralcare_user");
    if (stored) {
      setUser(JSON.parse(stored));
    }
    loadDemoUsers();
  }, []);

  async function loadDemoUsers() {
    try {
      const data = await getDemoUsers();
      setDemoUsers(data.users);
    } catch (err) {
      setError("Failed to connect to backend. Please ensure the server is running.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLogin(userId) {
    try {
      const actor = await demoLogin(userId);
      setUser(actor);
      localStorage.setItem("oralcare_user", JSON.stringify(actor));
    } catch (err) {
      setError(err.message);
    }
  }

  function handleLogout() {
    setUser(null);
    setActiveView("dashboard");
    localStorage.removeItem("oralcare_user");
  }

  function renderWorkspace() {
    if (activeView === "appointments") {
      return (
        <div className="dashboard">
          <header className="dashboard-header">
            <div className="header-info">
              <h1>Appointment Tracker</h1>
              <p>Requests, scheduling, doctor notes, and visit status</p>
            </div>
          </header>
          <AppointmentsPanel user={user} patientId={user.profile?.patient_id} />
        </div>
      );
    }
    if (user.user.role === "lab_technician") return <TechnicianDashboard user={user} />;
    if (user.user.role === "doctor") return <DoctorDashboard user={user} view={activeView} />;
    return <PatientDashboard user={user} view={activeView} />;
  }

  if (loading) return <div className="app-loader">Initializing clinical workspace...</div>;

  if (!user) {
    return (
      <div className="login-screen">
        <div className="login-card">
          <header>
            <Brain size={48} className="logo" />
            <h1>OralCare-AI</h1>
            <p>Multimodal Clinical Decision Support</p>
          </header>
          
          <div className="demo-login">
            <h3>Select a Demo User to Begin</h3>
            {error && <p className="error">{error}</p>}
            <div className="user-grid">
              {demoUsers.map(u => {
                const Icon = roleIcons[u.role] || UserRound;
                return (
                  <button key={u.user_id} onClick={() => handleLogin(u.user_id)} className="user-login-card">
                    <Icon size={24} />
                    <div className="info">
                      <strong>{u.name}</strong>
                      <span>{u.role.replaceAll("_", " ")}</span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
          
          <footer>
            <ShieldCheck size={14} />
            <span>Clinical workflow prototype</span>
          </footer>
        </div>
      </div>
    );
  }

  return (
    <main className="app-container">
      <nav className="side-nav">
        <div className="nav-brand">
          <Brain size={28} />
          <span>OralCare-AI</span>
        </div>
        <div className="nav-links">
          <div className="nav-section">MAIN</div>
          <button className={activeView === "dashboard" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("dashboard")}>
            <Activity size={18} />
            <span>Dashboard</span>
          </button>
          {user.user.role === "patient" && (
            <>
              <button className={activeView === "reports" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("reports")}>
                <ClipboardList size={18} />
                <span>Reports</span>
              </button>
            </>
          )}
          {user.user.role !== "lab_technician" && (
            <button className={activeView === "appointments" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("appointments")}>
              <CalendarClock size={18} />
              <span>Appointments</span>
            </button>
          )}
        </div>
        <div className="nav-user">
          <div className="user-info">
            <strong>{user.user.name}</strong>
            <span>{user.user.role.replaceAll("_", " ")}</span>
          </div>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <section className="content-area">
        <section className="disclaimer-strip">
          <AlertTriangle size={14} />
          Decision support only. All outputs require clinical validation.
        </section>

        {renderWorkspace()}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
