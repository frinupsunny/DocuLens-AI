"""
DocuLens AI - Chat Manager Module
Maintains conversation history, message structuring, source references,
and session reset controls.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional


class ChatMessage:
    """Represents a single message exchange in the chat session."""

    def __init__(
        self,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[str] = None
    ):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.sources = sources or []
        self.timestamp = timestamp or datetime.now().strftime("%I:%M %p")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }


class ChatManager:
    """
    Manages active session conversation history, export functions,
    and history resets.
    """

    def __init__(self):
        self._history: List[ChatMessage] = []

    @property
    def messages(self) -> List[ChatMessage]:
        return self._history

    def add_user_message(self, content: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=content)
        self._history.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessage:
        msg = ChatMessage(role="assistant", content=content, sources=sources)
        self._history.append(msg)
        return msg

    def clear(self) -> None:
        """Clear all messages in the active conversation."""
        self._history.clear()

    def get_total_messages(self) -> int:
        return len(self._history)

    def export_as_markdown(self) -> str:
        """Export the conversation as a formatted Markdown transcript."""
        if not self._history:
            return "No conversation history available."

        lines = ["# DocuLens AI - Conversation Transcript\n"]
        for msg in self._history:
            sender = "👤 **User**" if msg.role == "user" else "🤖 **DocuLens AI**"
            lines.append(f"### {sender} ({msg.timestamp})\n")
            lines.append(f"{msg.content}\n")
            if msg.sources:
                lines.append("**Sources Cited:**")
                for s in msg.sources:
                    lines.append(f"- *{s['source']}* (Page {s['page']})")
                lines.append("")
            lines.append("---\n")
        return "\n".join(lines)
