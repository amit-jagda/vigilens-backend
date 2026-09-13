import re
import logging
from typing import Optional, List, Dict, Any
import httpx
from configs.base import settings

logger = logging.getLogger(__name__)

class UnifiedLLMClient:
    """
    Unified Asynchronous LLM Client supporting:
    1. Cloud Groq (Llama 3.3 70B, Qwen 2.5 32B - fast & free tier)
    2. Cloud OpenAI (GPT-4o, GPT-4o-mini)
    3. Cloud Google Gemini (Gemini 2.0 Flash, Gemini 1.5 Flash)
    4. Cloud OpenRouter / DeepSeek / Custom OpenAI-compatible endpoints
    5. Local Ollama (qwen2.5-coder, llama3.1, mistral)
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or settings.LLM_MODEL
        self.base_url = base_url
        self.timeout = httpx.Timeout(settings.SQL_AGENT_TIMEOUT_SECONDS + 20.0, connect=5.0)

    def _resolve_provider_and_endpoint(self) -> tuple[str, str, str, Dict[str, str]]:
        """Resolves provider, base_url, model, and headers based on config and available API keys."""
        # Auto-detect if provider not explicitly set or set to generic
        if settings.GROQ_API_KEY and self.provider in ["groq", "auto", "default"]:
            return (
                "groq",
                "https://api.groq.com/openai/v1/chat/completions",
                self.model or "llama-3.3-70b-versatile",
                {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"},
            )
        elif settings.OPENAI_API_KEY and self.provider in ["openai", "auto", "default"]:
            base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
            return (
                "openai",
                f"{base}/chat/completions",
                self.model or "gpt-4o-mini",
                {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"},
            )
        elif settings.GEMINI_API_KEY and self.provider in ["gemini", "auto", "default"]:
            model_name = self.model or "gemini-2.0-flash"
            return (
                "gemini",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.GEMINI_API_KEY}",
                model_name,
                {"Content-Type": "application/json"},
            )
        elif self.provider == "ollama":
            base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
            return (
                "ollama",
                f"{base}/api/chat",
                self.model or settings.OLLAMA_MODEL or "qwen2.5-coder:latest",
                {"Content-Type": "application/json"},
            )
        else:
            # Fallback check for any configured key
            if settings.GROQ_API_KEY:
                return (
                    "groq",
                    "https://api.groq.com/openai/v1/chat/completions",
                    "llama-3.3-70b-versatile",
                    {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"},
                )
            if settings.OPENAI_API_KEY:
                return (
                    "openai",
                    "https://api.openai.com/v1/chat/completions",
                    "gpt-4o-mini",
                    {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"},
                )
            if settings.GEMINI_API_KEY:
                model_name = "gemini-2.0-flash"
                return (
                    "gemini",
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.GEMINI_API_KEY}",
                    model_name,
                    {"Content-Type": "application/json"},
                )
            
            # Default to Ollama local
            base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
            return (
                "ollama",
                f"{base}/api/chat",
                self.model or settings.OLLAMA_MODEL or "qwen2.5-coder:latest",
                {"Content-Type": "application/json"},
            )

    async def check_health(self) -> Dict[str, Any]:
        """Checks connection status and reports active engine details."""
        provider, endpoint, model, headers = self._resolve_provider_and_endpoint()

        if provider == "ollama":
            base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.get(f"{base}/api/tags")
                    if res.status_code == 200:
                        data = res.json()
                        models = [m.get("name") for m in data.get("models", [])]
                        return {
                            "status": "connected",
                            "provider": "ollama (local)",
                            "current_model": model,
                            "available_models": models,
                        }
            except Exception as e:
                return {
                    "status": "offline",
                    "provider": "ollama (local)",
                    "current_model": model,
                    "error": str(e),
                    "available_models": [],
                }

        # For cloud providers (Gemini, Groq, OpenAI)
        if provider == "gemini":
            has_key = bool(settings.GEMINI_API_KEY)
            models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            return {
                "status": "connected" if has_key else "needs_api_key",
                "provider": "GEMINI (Cloud)",
                "current_model": model if model in models else "gemini-2.0-flash",
                "available_models": models,
                "error": None if has_key else "GEMINI_API_KEY is not set in vigilens-backend/.env",
            }
        elif provider == "groq":
            has_key = bool(settings.GROQ_API_KEY)
            models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-2.5-coder-32b"]
            return {
                "status": "connected" if has_key else "needs_api_key",
                "provider": "GROQ (Cloud)",
                "current_model": model if model in models else "llama-3.3-70b-versatile",
                "available_models": models,
                "error": None if has_key else "GROQ_API_KEY is not set in vigilens-backend/.env",
            }
        elif provider == "openai":
            has_key = bool(settings.OPENAI_API_KEY)
            models = ["gpt-4o-mini", "gpt-4o"]
            return {
                "status": "connected" if has_key else "needs_api_key",
                "provider": "OPENAI (Cloud)",
                "current_model": model if model in models else "gpt-4o-mini",
                "available_models": models,
                "error": None if has_key else "OPENAI_API_KEY is not set in vigilens-backend/.env",
            }

        has_key = bool(settings.GROQ_API_KEY or settings.OPENAI_API_KEY or settings.GEMINI_API_KEY)
        return {
            "status": "connected" if has_key else "needs_api_key",
            "provider": f"{provider.upper()} (Cloud)",
            "current_model": model,
            "available_models": ["gemini-2.0-flash", "gemini-1.5-flash"],
            "error": None if has_key else "No API Key configured in .env",
        }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Sends chat request to active Cloud or Local LLM endpoint."""
        provider, endpoint, default_model, headers = self._resolve_provider_and_endpoint()
        target_model = model or default_model

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                if provider in ["groq", "openai", "openrouter"]:
                    payload = {
                        "model": target_model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    response = await client.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

                elif provider == "gemini":
                    # Convert OpenAI message format to Gemini format
                    gemini_contents = []
                    system_text = ""
                    for m in messages:
                        if m["role"] == "system":
                            system_text += m["content"] + "\n"
                        else:
                            role = "user" if m["role"] == "user" else "model"
                            gemini_contents.append({
                                "role": role,
                                "parts": [{"text": m["content"]}],
                            })

                    payload: Dict[str, Any] = {
                        "contents": gemini_contents,
                        "generationConfig": {"temperature": temperature},
                    }
                    if system_text:
                        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

                    response = await client.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        return "".join([p.get("text", "") for p in parts]).strip()
                    return ""

                elif provider == "ollama":
                    payload = {
                        "model": target_model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    }
                    response = await client.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("message", {}).get("content", "").strip()

            except Exception as exc:
                logger.error(f"LLM API Error ({provider}): {exc}")
                raise RuntimeError(f"LLM Request failed ({provider}): {exc}") from exc

        return ""

    @staticmethod
    def extract_sql_from_text(text: str) -> str:
        """Extracts clean SQL from model response."""
        if not text:
            return ""

        sql_block_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if sql_block_match:
            clean_sql = sql_block_match.group(1).strip()
        else:
            clean_sql = text.strip()

        lines = []
        for line in clean_sql.splitlines():
            line_str = line.strip()
            if line_str.startswith("--") or line_str.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()


# Backwards compatibility alias
OllamaClient = UnifiedLLMClient


def generate_fallback_sql(question: str, tenant_id: str) -> str:
    """Deterministic rule-based fallback query generator when offline/unreachable."""
    q = question.lower()

    if any(k in q for k in ["dwell", "longest", "stay", "duration"]):
        return f"""SELECT 
    i.id AS person_id,
    COALESCE(i.visitor_name, CONCAT(e.first_name, ' ', e.last_name), 'Unknown') AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    ROUND(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at)))) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ARRAY_AGG(DISTINCT pte.camera_name) AS cameras_visited
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
GROUP BY i.id, name, person_type
ORDER BY total_dwell_seconds DESC
LIMIT 10;"""

    if any(k in q for k in ["camera", "traffic", "busy", "busiest", "area", "zone"]):
        return f"""SELECT 
    pte.camera_name,
    COUNT(DISTINCT pte.identity_id) AS unique_visitors_count,
    COUNT(pte.id) AS total_events_count,
    ROUND(AVG(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at)))) AS avg_dwell_seconds
FROM person_timeline_events pte
WHERE pte.tenant_id = '{tenant_id}'
GROUP BY pte.camera_name
ORDER BY unique_visitors_count DESC
LIMIT 10;"""

    if any(k in q for k in ["employee", "staff", "worker"]):
        return f"""SELECT 
    e.id AS employee_id,
    CONCAT(e.first_name, ' ', e.last_name) AS name,
    e.employee_code,
    'employee' AS person_type,
    MIN(eal.employee_entry_timestamp) AS first_seen,
    MAX(eal.employee_exit_timestamp) AS last_seen,
    SUM(eal.occurrence_count) AS detection_count
FROM employees e
JOIN advanced_employee_attendance_logs eal ON eal.employee_id = e.id
WHERE e.tenant_id = '{tenant_id}'
GROUP BY e.id, e.first_name, e.last_name, e.employee_code
ORDER BY first_seen DESC
LIMIT 25;"""

    return f"""SELECT 
    i.id AS person_id,
    COALESCE(i.visitor_name, CONCAT(e.first_name, ' ', e.last_name), 'Unknown') AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
GROUP BY i.id, name, person_type
ORDER BY total_dwell_seconds DESC
LIMIT 25;"""
