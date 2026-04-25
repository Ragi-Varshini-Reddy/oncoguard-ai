import React, { useEffect, useState } from "react";
import { CalendarClock, CheckCircle2, RefreshCw, Send } from "lucide-react";
import { getAppointments, requestAppointment, updateAppointment } from "../services/api";

export default function AppointmentsPanel({ user, patientId }) {
  const [appointments, setAppointments] = useState([]);
  const [status, setStatus] = useState("");
  const [form, setForm] = useState({
    requested_date: "",
    issue: "",
    reason: "",
  });
  const isPatient = user?.user?.role === "patient";
  const isDoctor = user?.user?.role === "doctor";

  useEffect(() => {
    loadAppointments();
  }, [user?.user?.user_id]);

  async function loadAppointments() {
    try {
      const data = await getAppointments();
      setAppointments(data.appointments || []);
    } catch (err) {
      setStatus(err.message || "Could not load appointments.");
    }
  }

  async function submitRequest(event) {
    event.preventDefault();
    if (!patientId) return;
    setStatus("Sending appointment request...");
    try {
      await requestAppointment(patientId, form);
      setForm({ requested_date: "", issue: "", reason: "" });
      setStatus("Appointment request sent.");
      await loadAppointments();
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function changeStatus(appointmentId, nextStatus, currentDate = "", notes = "") {
    setStatus("Updating appointment...");
    try {
      await updateAppointment(appointmentId, {
        status: nextStatus,
        requested_date: currentDate || null,
        doctor_notes: notes || null,
      });
      setStatus("Appointment updated.");
      await loadAppointments();
    } catch (err) {
      setStatus(err.message);
    }
  }

  return (
    <section className="appointments-panel panel">
      <div className="panel-head">
        <CalendarClock size={20} />
        <h3>Appointments</h3>
        <button className="secondary compact" onClick={loadAppointments}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {isPatient && (
        <form className="appointment-form" onSubmit={submitRequest}>
          <input
            type="datetime-local"
            value={form.requested_date}
            onChange={event => setForm(prev => ({ ...prev, requested_date: event.target.value }))}
            aria-label="Preferred appointment date and time"
            required
          />
          <input
            value={form.issue}
            onChange={event => setForm(prev => ({ ...prev, issue: event.target.value }))}
            placeholder="Main issue"
            required
          />
          <textarea
            value={form.reason}
            onChange={event => setForm(prev => ({ ...prev, reason: event.target.value }))}
            placeholder="Reason for appointment"
            required
          />
          <button className="primary" type="submit">
            <Send size={16} /> Request Appointment
          </button>
        </form>
      )}

      <div className="appointment-list">
        {appointments.map(item => (
          <AppointmentRow
            key={item.appointment_id}
            item={item}
            isDoctor={isDoctor}
            onStatus={changeStatus}
          />
        ))}
        {appointments.length === 0 && <p className="muted">No appointments tracked yet.</p>}
      </div>
      {status && <p className="muted">{status}</p>}
    </section>
  );
}

function AppointmentRow({ item, isDoctor, onStatus }) {
  const [date, setDate] = useState(item.requested_date || "");
  const [notes, setNotes] = useState(item.doctor_notes || "");
  return (
    <div className="appointment-row">
      <div>
        <strong>{item.issue}</strong>
        <span>{item.patient_id} · {item.requested_date}</span>
        <small>{item.reason}</small>
        {item.doctor_notes && <small>Doctor notes: {item.doctor_notes}</small>}
      </div>
      <span className={`status-badge ${item.status === "completed" ? "available" : ""}`}>{item.status}</span>
      {isDoctor && (
        <div className="appointment-controls">
          <input type="datetime-local" value={date} onChange={event => setDate(event.target.value)} aria-label="Scheduled appointment date and time" />
          <input value={notes} onChange={event => setNotes(event.target.value)} placeholder="Doctor notes" />
          <button className="secondary compact" onClick={() => onStatus(item.appointment_id, "scheduled", date, notes)}>
            <CalendarClock size={14} /> Schedule
          </button>
          <button className="primary compact" onClick={() => onStatus(item.appointment_id, "completed", date, notes)}>
            <CheckCircle2 size={14} /> Complete
          </button>
        </div>
      )}
    </div>
  );
}
