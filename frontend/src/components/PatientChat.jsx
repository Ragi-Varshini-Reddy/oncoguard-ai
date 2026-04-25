import React, { useState, useEffect, useRef } from "react";
import { Send, Bot, User, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { sendPatientChat } from "../services/api";

export default function PatientChat({ patientId, latestRun }) {
  const storageKey = `oralcare_chat_session_${patientId}`;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    setSessionId(localStorage.getItem(storageKey));
  }, [storageKey]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const moduleOutputs = latestRun?.module_outputs || [];
      const fusionOutput = latestRun?.fusion_output || null;
      
      const response = await sendPatientChat(
        patientId, 
        sessionId, 
        userMsg, 
        moduleOutputs, 
        fusionOutput
      );
      
      if (response.messages?.length) {
        setMessages(response.messages);
      } else {
        setMessages(prev => [...prev, { role: "assistant", content: response.answer }]);
      }
      setSessionId(response.session_id);
      localStorage.setItem(storageKey, response.session_id);
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", content: "I'm sorry, I encountered an error connecting to my clinical knowledge base. Please try again in a moment." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="patient-chat">
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-empty-state">
            <Bot size={18} />
            <span>Ask OralCare-AI about your processed records, doctor, risk trend, uploads, or report.</span>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message-wrapper ${m.role}`}>
            <div className="avatar">
              {m.role === "assistant" ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className="message-content">
              {m.role === "assistant" ? (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-wrapper assistant">
            <div className="avatar"><Bot size={16} /></div>
            <div className="message-content typing">
              <Sparkles size={14} className="spin" /> Thinking...
            </div>
          </div>
        )}
      </div>

      <form className="chat-input" onSubmit={handleSend}>
        <input 
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about your records..."
          disabled={loading}
        />
        <button type="submit" disabled={!input.trim() || loading}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
