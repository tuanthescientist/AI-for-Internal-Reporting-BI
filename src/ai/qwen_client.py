"""
QwenAIClient — core Groq/Qwen-3 streaming client.

All AI interactions in the platform are routed through this class.
Author: Tuan Tran
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Dict, Any, Optional

from groq import Groq

from src.config.settings import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Groq client (lazily initialised)
# ─────────────────────────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not Config.GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        _groq_client = Groq(api_key=Config.GROQ_API_KEY)
    return _groq_client


# ─────────────────────────────────────────────────────────────────────────────
class QwenAIClient:
    """
    Stateful AI client wrapping the Groq → Qwen 3 32B model.

    Maintains per-session conversation history for multi-turn dialogue.
    All public methods return generators that yield text chunks (streamed).

    Usage
    -----
    client = QwenAIClient()
    for chunk in client.chat("What drove the Q3 revenue decline?"):
        print(chunk, end="", flush=True)
    """

    # ── System prompts ────────────────────────────────────────
    _SYSTEM_BI_ANALYST = (
        "You are a senior Business Intelligence analyst and data storyteller. "
        "You work with internal enterprise reporting across Sales, Finance, HR, "
        "and Operations. Your responses are concise, professional, and action-oriented. "
        "Always reference specific numbers from the provided data context. "
        "Never fabricate figures not present in the context. "
        "Format responses with clear headers (##), bullet points, and tables where appropriate."
    )

    _SYSTEM_REPORT_WRITER = (
        "You are a professional business report writer for Fortune 500 internal reporting. "
        "Write in a formal, executive-ready style. Structure every report with: "
        "Executive Summary, Key Findings, Detailed Analysis, Risks & Opportunities, "
        "and Strategic Recommendations. Use markdown formatting."
    )

    _SYSTEM_ANOMALY_ANALYST = (
        "You are an AI-powered anomaly detection specialist. "
        "Your task is to identify statistically unusual patterns in business metrics "
        "and explain them in plain business language. "
        "Assign a severity: [LOW / MEDIUM / HIGH / CRITICAL]. "
        "For each anomaly, provide: What happened, Why it matters, Recommended action."
    )

    _SYSTEM_INSIGHT_ENGINE = (
        "You are a KPI insight engine for internal BI dashboards. "
        "Generate concise, punchy insights (2–4 sentences max per KPI). "
        "Be direct and data-driven. Highlight trend direction, magnitude, and implication."
    )

    def __init__(self) -> None:
        self._history: List[Dict[str, str]] = []

    # ── Core streaming method ─────────────────────────────────
    def _stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = Config.AI_TEMPERATURE,
        max_tokens: int = Config.AI_MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """Low-level streaming call to Groq API."""
        client = _get_groq_client()
        try:
            completion = client.chat.completions.create(
                model=Config.QWEN_MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=Config.AI_TOP_P,
                reasoning_effort=Config.AI_REASONING_EFFORT,
                stream=True,
                stop=None,
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            yield f"\n\n⚠️ AI Error: {exc}"

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str,
        include_history: bool = False,
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if include_history:
            msgs.extend(self._history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    # ── Public API ────────────────────────────────────────────
    def chat(
        self, message: str, data_context: str = ""
    ) -> Generator[str, None, None]:
        """
        General-purpose conversational BI chat.
        Maintains conversation history for multi-turn dialogue.
        """
        content = message
        if data_context:
            content = f"**Data Context:**\n{data_context}\n\n**Question:** {message}"

        msgs = self._build_messages(
            content, self._SYSTEM_BI_ANALYST, include_history=True
        )

        full_response = ""
        for chunk in self._stream(msgs):
            full_response += chunk
            yield chunk

        # Update history
        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "assistant", "content": full_response})

    def generate_executive_report(
        self, data_summary: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate a structured executive report from a multi-domain data summary."""
        prompt = (
            "Generate a comprehensive internal executive report for this period's data.\n\n"
            f"**Business Data Summary:**\n```json\n{json.dumps(data_summary, indent=2, default=str)}\n```\n\n"
            "Include ALL sections: Executive Summary, Sales Performance, Financial Health, "
            "HR Metrics, Operations KPIs, Key Risks, and Strategic Recommendations."
        )
        msgs = self._build_messages(prompt, self._SYSTEM_REPORT_WRITER)
        yield from self._stream(msgs, temperature=0.5)

    def generate_department_report(
        self, department: str, department_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate a focused department-level report."""
        prompt = (
            f"Generate a detailed internal report for the **{department}** department.\n\n"
            f"**{department} Data:**\n```json\n"
            f"{json.dumps(department_data, indent=2, default=str)}\n```\n\n"
            "Focus on: performance vs targets, key trends, concerns, and recommended actions."
        )
        msgs = self._build_messages(prompt, self._SYSTEM_REPORT_WRITER)
        yield from self._stream(msgs, temperature=0.5)

    def explain_anomalies(
        self, anomalies: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Provide AI narrative explanation for detected statistical anomalies."""
        prompt = (
            "Analyse the following detected anomalies in our business metrics and "
            "provide clear explanations, business impact, and recommended actions.\n\n"
            f"**Detected Anomalies:**\n```json\n"
            f"{json.dumps(anomalies, indent=2, default=str)}\n```"
        )
        msgs = self._build_messages(prompt, self._SYSTEM_ANOMALY_ANALYST)
        yield from self._stream(msgs, temperature=0.4)

    def get_kpi_insights(
        self, kpi_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate concise AI-powered insights for a set of KPIs."""
        prompt = (
            "Generate crisp, executive-level insights for the following KPI metrics.\n\n"
            f"**KPIs:**\n```json\n{json.dumps(kpi_data, indent=2, default=str)}\n```\n\n"
            "For each KPI, provide a 1–2 sentence insight highlighting trend and implication."
        )
        msgs = self._build_messages(prompt, self._SYSTEM_INSIGHT_ENGINE)
        yield from self._stream(msgs, temperature=0.6)

    def natural_language_query(
        self, question: str, data_context: str
    ) -> Generator[str, None, None]:
        """Answer a natural language business question using provided data context."""
        prompt = (
            f"**Business Data:**\n{data_context}\n\n"
            f"**Question:** {question}\n\n"
            "Answer using only information available in the data above. "
            "Be specific with numbers. If unsure, say so clearly."
        )
        msgs = self._build_messages(prompt, self._SYSTEM_BI_ANALYST)
        yield from self._stream(msgs)

    # ── Utility ───────────────────────────────────────────────
    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._history.clear()

    @property
    def history_length(self) -> int:
        return len(self._history)
