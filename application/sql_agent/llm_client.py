import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from configs.base import settings

logger = logging.getLogger(__name__)

# Try importing httpx, fallback to urllib if not installed
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import urllib.request
    import urllib.error

class UnifiedLLMClient:
    """
    Unified Resilient Asynchronous LLM Client supporting:
    1. Cloud Google Gemini (gemini-flash-latest, gemini-2.5-flash-lite, gemini-pro-latest, gemini-2.5-pro)
    2. Cloud Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen-2.5-coder-32b)
    3. Cloud OpenAI (gpt-4o, gpt-4o-mini)
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
        self.timeout_seconds = settings.SQL_AGENT_TIMEOUT_SECONDS + 20.0

    def _map_gemini_model(self, model_name: Optional[str]) -> str:
        """Maps requested or deprecated Gemini model names to active endpoints."""
        if not model_name:
            return "gemini-flash-latest"
        m = model_name.strip()
        if m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash", "flash"]:
            return "gemini-flash-latest"
        if m in ["gemini-pro", "gemini-1.5-pro", "pro"]:
            return "gemini-pro-latest"
        return m

    def _resolve_provider_and_endpoint(self) -> tuple[str, str, str, Dict[str, str]]:
        """Resolves provider, base_url, model, and headers based on config and available API keys."""
        # Auto-detect if provider not explicitly set or set to generic
        if settings.GEMINI_API_KEY and self.provider in ["gemini", "auto", "default"]:
            model_name = self._map_gemini_model(self.model)
            return (
                "gemini",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.GEMINI_API_KEY}",
                model_name,
                {"Content-Type": "application/json"},
            )
        elif settings.GROQ_API_KEY and self.provider in ["groq", "auto", "default"]:
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
            if settings.GEMINI_API_KEY:
                model_name = self._map_gemini_model(self.model)
                return (
                    "gemini",
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.GEMINI_API_KEY}",
                    model_name,
                    {"Content-Type": "application/json"},
                )
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
            
            # Default to Ollama local
            base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
            return (
                "ollama",
                f"{base}/api/chat",
                self.model or settings.OLLAMA_MODEL or "qwen2.5-coder:latest",
                {"Content-Type": "application/json"},
            )

    async def _post_json(self, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Performs async HTTP POST with resilient fallback to urllib."""
        if HAS_HTTPX:
            try:
                timeout = httpx.Timeout(self.timeout_seconds, connect=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                logger.debug(f"httpx POST failed ({exc}), attempting urllib fallback...")

        # Fallback to standard library urllib executed in worker thread
        def _urllib_post():
            import urllib.request
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))

        return await asyncio.to_thread(_urllib_post)

    async def check_health(self) -> Dict[str, Any]:
        """Checks connection status and reports active engine details."""
        provider, endpoint, model, headers = self._resolve_provider_and_endpoint()

        if provider == "gemini":
            has_key = bool(settings.GEMINI_API_KEY)
            models = ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-pro-latest", "gemini-2.5-pro"]
            return {
                "status": "connected" if has_key else "needs_api_key",
                "provider": "Google Gemini (Cloud)",
                "current_model": model,
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
                "provider": "OpenAI (Cloud)",
                "current_model": model if model in models else "gpt-4o-mini",
                "available_models": models,
                "error": None if has_key else "OPENAI_API_KEY is not set in vigilens-backend/.env",
            }
        elif provider == "ollama":
            base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
            try:
                if HAS_HTTPX:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        res = await client.get(f"{base}/api/tags")
                        if res.status_code == 200:
                            data = res.json()
                            models = [m.get("name") for m in data.get("models", [])]
                            return {
                                "status": "connected",
                                "provider": "Ollama (Local)",
                                "current_model": model,
                                "available_models": models,
                            }
            except Exception as e:
                return {
                    "status": "offline",
                    "provider": "Ollama (Local)",
                    "current_model": model,
                    "error": str(e),
                    "available_models": [],
                }

        has_key = bool(settings.GROQ_API_KEY or settings.OPENAI_API_KEY or settings.GEMINI_API_KEY)
        return {
            "status": "connected" if has_key else "needs_api_key",
            "provider": f"{provider.upper()} (Cloud)",
            "current_model": model,
            "available_models": ["gemini-flash-latest", "gemini-2.5-flash-lite"],
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
        target_model = self._map_gemini_model(model) if provider == "gemini" else (model or default_model)

        # Candidate models for failover
        candidate_models = [target_model]
        if provider == "gemini":
            for fallback_m in ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-flash-latest"]:
                if fallback_m not in candidate_models:
                    candidate_models.append(fallback_m)

        last_exc = None
        for current_m in candidate_models:
            if provider == "gemini":
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{current_m}:generateContent?key={settings.GEMINI_API_KEY}"

            try:
                if provider in ["groq", "openai", "openrouter"]:
                    payload = {
                        "model": current_m,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    data = await self._post_json(endpoint, payload, headers)
                    return data["choices"][0]["message"]["content"].strip()

                elif provider == "gemini":
                    gemini_contents = []
                    system_text = ""
                    for m in messages:
                        if m.get("role") == "system":
                            system_text += m.get("content", "") + "\n"
                        else:
                            role = "user" if m.get("role") == "user" else "model"
                            gemini_contents.append({
                                "role": role,
                                "parts": [{"text": m.get("content", "")}],
                            })

                    payload = {
                        "contents": gemini_contents,
                        "generationConfig": {"temperature": temperature},
                    }
                    if system_text:
                        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

                    data = await self._post_json(endpoint, payload, headers)
                    
                    # Check for valid Gemini response structure
                    if "candidates" in data and data["candidates"]:
                        parts = data["candidates"][0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
                    if "error" in data:
                        raise Exception(f"Gemini API Error: {data['error'].get('message', data['error'])}")
                    raise Exception(f"Unexpected Gemini response: {data}")

                elif provider == "ollama":
                    payload = {
                        "model": current_m,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    }
                    data = await self._post_json(endpoint, payload, headers)
                    return data["message"]["content"].strip()

            except Exception as exc:
                last_exc = exc
                logger.warning(f"LLM attempt failed on {provider}/{current_m}: {exc}. Trying next candidate...")
                continue

        raise Exception(f"All LLM candidates failed for {provider}: {last_exc}")

    @staticmethod
    def extract_sql_from_text(text: str) -> str:
        """Extracts SQL query from LLM output, discarding markdown fences and explanations."""
        if not text:
            return ""

        # Check for ```sql ... ``` or ``` ... ``` code blocks
        sql_block_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if sql_block_match:
            candidate = sql_block_match.group(1).strip()
            if candidate.lower().startswith("select") or candidate.lower().startswith("with"):
                return candidate

        # Look for first SELECT statement
        select_match = re.search(r"(SELECT\s+[\s\S]*?;?)", text, re.IGNORECASE)
        if select_match:
            candidate = select_match.group(1).strip()
            # Clean up trailing markdown artifacts if any
            candidate = candidate.split("```")[0].strip()
            return candidate

        return text.strip()


# Backwards compatibility alias
OllamaClient = UnifiedLLMClient


def generate_fallback_sql(question: str, tenant_id: str) -> str:
    """
    Synthesizes domain-informed fallback SQL query for common CCTV queries
    when the LLM model is unavailable or rate-limited.
    """
    q = question.lower().strip()

    # Time window detection
    time_filter = ""
    if "yesterday" in q:
        time_filter = "AND pte.started_at >= CURRENT_DATE - INTERVAL '1 day' AND pte.started_at < CURRENT_DATE"
    elif "today" in q:
        time_filter = "AND pte.started_at >= CURRENT_DATE"
    elif "week" in q or "7 days" in q:
        time_filter = "AND pte.started_at >= NOW() - INTERVAL '7 days'"
    elif "hour" in q:
        time_filter = "AND pte.started_at >= NOW() - INTERVAL '1 hour'"

    # Common name expression: ensures non-null, non-empty, clean names
    name_sql = (
        "COALESCE(NULLIF(TRIM(i.visitor_name), ''), "
        "NULLIF(TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))), ''), "
        "CONCAT('Visitor #', SUBSTRING(i.id::text, 1, 6)), 'Unknown Visitor')"
    )

    # 1. Check if user wants ALL EMPLOYEES / STAFF DIRECTORY (general list query)
    is_employee_list = any(phrase in q for phrase in [
        "list down the employees", "list employees", "list of employees",
        "show all employees", "show employees", "all employees",
        "who are the employees", "employee directory", "staff directory",
        "all staff", "show staff", "list staff", "enrolled employees",
        "registered employees", "show all staff members", "all workers"
    ]) or (
        ("employee" in q or "employees" in q or "staff" in q or "workers" in q)
        and any(w in q for w in ["list", "all", "directory", "show", "get", "everyone", "who are", "roster"])
        and not any(w in q for w in ["where is", "details for", "dwell of", "dwell time of", "named", "specific"])
    )

    if is_employee_list:
        return f"""SELECT 
    e.id AS person_id,
    TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))) AS name,
    e.employee_code,
    e.photo_path,
    'employee' AS person_type,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen
FROM employees e
LEFT JOIN advanced_person_identities i ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON (pte.employee_id = e.id OR pte.identity_id = i.id)
WHERE e.tenant_id = '{tenant_id}'
GROUP BY e.id, e.first_name, e.last_name, e.employee_code, e.photo_path
ORDER BY e.first_name ASC, e.last_name ASC
LIMIT 50;"""

    # 2. Object / Belongings Query
    if any(k in q for k in ["backpack", "laptop", "bag", "phone", "object", "carrying", "suitcase", "item", "belonging"]):
        obj_match = ""
        for obj in ["backpack", "laptop", "bottle", "handbag", "suitcase", "cell phone"]:
            if obj in q or (obj == "cell phone" and "phone" in q) or (obj == "handbag" and "bag" in q):
                obj_match = f"AND pte.associated_objects::text ILIKE '%{obj.replace('cell phone', 'phone')}%'"
                break

        return f"""SELECT 
    i.id AS person_id,
    {name_sql} AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    pte.camera_name,
    pte.associated_objects,
    pte.started_at,
    pte.ended_at
FROM person_timeline_events pte
JOIN advanced_person_identities i ON pte.identity_id = i.id
LEFT JOIN employees e ON i.employee_id = e.id
WHERE pte.tenant_id = '{tenant_id}'
  AND pte.associated_objects IS NOT NULL
  AND jsonb_array_length(pte.associated_objects) > 0
  {obj_match}
  {time_filter}
ORDER BY pte.started_at DESC
LIMIT 25;"""

    # 3. Camera Traffic / Busiest Area
    if any(k in q for k in ["camera", "traffic", "busy", "busiest", "area", "zone", "location", "hotspot", "foot traffic"]):
        return f"""SELECT 
    pte.camera_name,
    COUNT(DISTINCT pte.identity_id) AS unique_visitors_count,
    COUNT(pte.id) AS total_events_count,
    ROUND(COALESCE(AVG(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS avg_dwell_seconds
FROM person_timeline_events pte
WHERE pte.tenant_id = '{tenant_id}'
  {time_filter}
GROUP BY pte.camera_name
ORDER BY unique_visitors_count DESC
LIMIT 10;"""

    # 4. Specific Person / Employee Lookup by Explicit Name or Code
    # Look for name patterns: "where is Kinjal", "dwell time of Kinjal", "show details for Alex", "EMP002"
    specific_patterns = [
        r"(?:where is|find|details for|info on|about|who is|look for)\s+([a-zA-Z0-9]+)",
        r"(?:dwell time of|dwell of|time spent by)\s+([a-zA-Z0-9]+)",
        r"([a-zA-Z0-9]+)'s\s+(?:dwell|journey|path|location|photo|details|attendance|time)",
        r"\b(emp\d+)\b",
    ]
    candidate_name = ""
    for pat in specific_patterns:
        m = re.search(pat, q)
        if m:
            cand = m.group(1).strip()
            if cand.lower() not in ["someone", "anyone", "person", "visitor", "employee", "employees", "staff", "the", "a", "who", "all", "down"]:
                candidate_name = cand
                break

    if candidate_name:
        return f"""SELECT 
    e.id AS person_id,
    TRIM(CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, ''))) AS name,
    e.employee_code,
    e.photo_path,
    'employee' AS person_type,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen
FROM employees e
LEFT JOIN advanced_person_identities i ON i.employee_id = e.id
LEFT JOIN person_timeline_events pte ON (pte.employee_id = e.id OR pte.identity_id = i.id)
WHERE e.tenant_id = '{tenant_id}'
  AND (
    e.first_name ILIKE '%{candidate_name}%' 
    OR e.last_name ILIKE '%{candidate_name}%' 
    OR CONCAT(e.first_name, ' ', e.last_name) ILIKE '%{candidate_name}%' 
    OR e.employee_code ILIKE '%{candidate_name}%'
  )
GROUP BY e.id, e.first_name, e.last_name, e.employee_code, e.photo_path
ORDER BY total_dwell_seconds DESC;"""

    # 5. Visitor Specific (who visited, recent guests)
    if any(k in q for k in ["visitor", "visitors", "guest", "guests", "who visited"]):
        return f"""SELECT 
    i.id AS person_id,
    {name_sql} AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
  {time_filter}
GROUP BY i.id, i.visitor_name, e.first_name, e.last_name, e.photo_path, pte.entry_crop_path, i.is_employee
ORDER BY total_dwell_seconds DESC
LIMIT 25;"""

    # 6. General Dwell Time / Longest Stay / General Surveillance Query
    return f"""SELECT 
    i.id AS person_id,
    {name_sql} AS name,
    CASE WHEN i.is_employee THEN 'employee' ELSE 'visitor' END AS person_type,
    COALESCE(e.photo_path, pte.entry_crop_path) AS photo_path,
    COUNT(DISTINCT pte.camera_name) AS camera_stops_count,
    ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (pte.ended_at - pte.started_at))), 0)) AS total_dwell_seconds,
    MIN(pte.started_at) AS first_seen,
    MAX(pte.ended_at) AS last_seen,
    ARRAY_AGG(DISTINCT pte.camera_name) FILTER (WHERE pte.camera_name IS NOT NULL) AS cameras_visited
FROM advanced_person_identities i
LEFT JOIN employees e ON i.employee_id = e.id
JOIN person_timeline_events pte ON pte.identity_id = i.id
WHERE i.tenant_id = '{tenant_id}'
  {time_filter}
GROUP BY i.id, i.visitor_name, e.first_name, e.last_name, e.photo_path, pte.entry_crop_path, i.is_employee
ORDER BY total_dwell_seconds DESC
LIMIT 25;"""
