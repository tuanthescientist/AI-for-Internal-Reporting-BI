"""
QwenAIClient — core Groq/Qwen-3 streaming client.

All AI interactions in the platform are routed through this class.
Supports automatic language detection: Qwen responds in the same
language the user writes in (Vietnamese ↔ English).

Author: Tuan Tran
"""

from __future__ import annotations

import json
import logging
import re
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
# Language detection (lightweight, no external library needed)
# ─────────────────────────────────────────────────────────────────────────────
_VI_PATTERN = re.compile(
    r"[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]",
    re.UNICODE,
)

_VI_KEYWORDS = re.compile(
    r"\b(là|và|của|không|có|cho|với|được|một|các|này|đó|bạn|tôi|"
    r"dữ liệu|doanh thu|báo cáo|phân tích|tăng|giảm|so sánh|"
    r"tháng|quý|năm|kết quả|hiệu suất|nhân viên|chi phí|lợi nhuận)\b",
    re.IGNORECASE | re.UNICODE,
)


def detect_language(text: str) -> str:
    """
    Returns 'vi' if the text is Vietnamese, else 'en'.
    Uses Unicode diacritic detection + common Vietnamese keyword matching.
    """
    if not text or not text.strip():
        return "en"
    vi_chars = len(_VI_PATTERN.findall(text))
    vi_kw    = len(_VI_KEYWORDS.findall(text))
    # Confident Vietnamese if ≥3 accented chars OR ≥2 Vietnamese keywords
    if vi_chars >= 3 or vi_kw >= 2:
        return "vi"
    return "en"


def _lang_instruction(lang: str) -> str:
    if lang == "vi":
        return (
            "\n\nQUAN TRỌNG: Người dùng đang viết bằng tiếng Việt. "
            "Hãy trả lời HOÀN TOÀN bằng tiếng Việt. "
            "Sử dụng thuật ngữ kinh doanh chuẩn tiếng Việt. "
            "Số tiền: định dạng VND hoặc USD rõ ràng."
        )
    return (
        "\n\nIMPORTANT: Respond entirely in English. "
        "Use standard business terminology."
    )


# ─────────────────────────────────────────────────────────────────────────────
class QwenAIClient:
    """
    Stateful AI client wrapping the Groq → Qwen 3 32B model.

    Key features:
    - Maintains per-session conversation history for multi-turn dialogue.
    - Auto-detects user language and instructs Qwen to respond in kind.
    - All public methods return generators that yield text chunks (streamed).

    Usage
    -----
    client = QwenAIClient()
    for chunk in client.chat("Doanh thu tháng này thế nào?"):
        print(chunk, end="", flush=True)
    """

    # ── Base system prompts (language suffix appended dynamically) ──────────
    _BASE_BI_ANALYST = (
        "You are a senior Business Intelligence analyst. Be concise—aim for 150 words max. "
        "Reference specific numbers from data only. Never fabricate figures. "
        "Use bullet points. Professional tone."
    )

    _BASE_REPORT_WRITER = (
        "You are an executive business report writer. Be concise—500 words max. "
        "Structure: Executive Summary (3 lines), Key Findings (bullets), Action Items. "
        "Data-driven, professional tone."
    )

    _BASE_ANOMALY_ANALYST = (
        "Explain each anomaly in 2-3 sentences: impact + root cause + action. "
        "Severity: LOW/MEDIUM/HIGH/CRITICAL. Concise, data-focused."
    )

    _BASE_INSIGHT_ENGINE = (
        "Generate 1-2 sentence insights per KPI: trend + implication. Concise, data-driven."
    )

    def __init__(self) -> None:
        self._history: List[Dict[str, str]] = []
        # Track the last detected language so action buttons stay consistent
        self._last_lang: str = "en"

    # ── Helpers ───────────────────────────────────────────────
    def _system(self, base: str, lang: str) -> str:
        return base + _lang_instruction(lang)

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
        Auto-detects language and maintains conversation history.
        """
        lang = detect_language(message)
        self._last_lang = lang

        content = message
        if data_context:
            content = f"Data Context:\n{data_context}\n\nQuestion: {message}"

        msgs = self._build_messages(
            content,
            self._system(self._BASE_BI_ANALYST, lang),
            include_history=True,
        )

        full_response = ""
        for chunk in self._stream(msgs):
            full_response += chunk
            yield chunk

        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "assistant", "content": full_response})

    def generate_executive_report(
        self, data_summary: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate a structured executive report using last detected language."""
        lang   = self._last_lang
        prompt = (
            "Generate a concise internal executive report.\n\n"
            f"Data:\n{json.dumps(data_summary, separators=(',', ':'), default=str)}\n\n"
            "Sections: Executive Summary, Key Findings, Risks, Recommendations."
        )
        msgs = self._build_messages(prompt, self._system(self._BASE_REPORT_WRITER, lang))
        yield from self._stream(msgs, temperature=0.5)

    def generate_department_report(
        self, department: str, department_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate a focused department-level report."""
        lang   = self._last_lang
        prompt = (
            f"Generate a concise internal report for {department}.\n\n"
            f"Data:\n{json.dumps(department_data, separators=(',', ':'), default=str)}\n\n"
            "Cover: performance vs targets, key trends, concerns, actions."
        )
        msgs = self._build_messages(prompt, self._system(self._BASE_REPORT_WRITER, lang))
        yield from self._stream(msgs, temperature=0.5)

    def explain_anomalies(
        self, anomalies: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """AI narrative explanation for detected anomalies."""
        lang   = self._last_lang
        prompt = (
            "Explain these business metric anomalies.\n\n"
            f"Anomalies:\n{json.dumps(anomalies, separators=(',', ':'), default=str)}"
        )
        msgs = self._build_messages(prompt, self._system(self._BASE_ANOMALY_ANALYST, lang))
        yield from self._stream(msgs, temperature=0.4)

    def get_kpi_insights(
        self, kpi_data: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate concise AI insights for a set of KPIs."""
        lang   = self._last_lang
        prompt = (
            "Give 1-2 sentence insights for each KPI.\n\n"
            f"KPIs:\n{json.dumps(kpi_data, separators=(',', ':'), default=str)}"
        )
        msgs = self._build_messages(prompt, self._system(self._BASE_INSIGHT_ENGINE, lang))
        yield from self._stream(msgs, temperature=0.6)

    def natural_language_query(
        self, question: str, data_context: str
    ) -> Generator[str, None, None]:
        """Answer a natural language business question."""
        lang   = detect_language(question)
        self._last_lang = lang
        prompt = (
            f"Data:\n{data_context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only data above. Be specific with numbers."
        )
        msgs = self._build_messages(prompt, self._system(self._BASE_BI_ANALYST, lang))
        yield from self._stream(msgs)

    def set_language(self, lang: str) -> None:
        """Manually override language ('en' or 'vi')."""
        self._last_lang = lang if lang in ("en", "vi") else "en"

    # ── Utility ───────────────────────────────────────────────
    def clear_history(self) -> None:
        self._history.clear()

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def current_language(self) -> str:
        return self._last_lang
