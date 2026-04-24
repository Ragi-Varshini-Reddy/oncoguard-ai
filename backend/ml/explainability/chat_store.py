"""Small in-memory chat session store for the demo backend."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    session_id: str
    patient_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class ChatStore:
    """Stores compact rolling chat history in memory.

    This is intentionally lightweight for hackathon/demo use. It stores only
    user/assistant text turns, not embeddings, uploaded files, or full artifacts.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, patient_id: str, session_id: str | None = None) -> ChatSession:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if session.patient_id == patient_id:
                return session
        new_session = ChatSession(session_id=session_id or str(uuid.uuid4()), patient_id=patient_id)
        self._sessions[new_session.session_id] = new_session
        return new_session

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> ChatSession:
        session = self._sessions[session_id]
        session.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        session.messages = session.messages[-self.max_turns * 2 :]
        session.updated_at = time.time()
        return session
