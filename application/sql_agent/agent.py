import logging
import json
import re
import difflib
from typing import Dict, Any, List, Optional, TYPE_CHECKING

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import text
except ImportError:
    AsyncSession = Any
    text = lambda s: s

from configs.base import settings
from application.sql_agent.schema_context import build_system_prompt
from application.sql_agent.llm_client import OllamaClient, generate_fallback_sql

try:
    from application.sql_agent.executor import execute_query_safe, SQLExecutorError
except ImportError:
    SQLExecutorError = Exception
    async def execute_query_safe(db: Any, sql_query: str, tenant_id: str):
        return [], 0.0

logger = logging.getLogger(__name__)

def format_dwell_time(seconds: int) -> str:
    """Formats duration in seconds into human-readable notation (e.g. 185s -> '3m 05s')."""
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    mins = seconds // 60
    rem_secs = seconds % 60
    if mins < 60:
        return f"{mins}m {rem_secs:02d}s" if rem_secs else f"{mins}m"
    hrs = mins // 60
    rem_mins = mins % 60
    return f"{hrs}h {rem_mins:02d}m {rem_secs:02d}s"


class SQLAgent:
    """
    Autonomous Spatio-Temporal Intelligence & SQL Agent for Vigilens.
    Features:
    1. Intent Classification (Conversational & Guidance vs Spatio-Temporal SQL Analytics)
    2. Fuzzy Name & Spelling Mistake Tolerance over Enrolled Staff & Known Visitors
    3. Multi-Turn Context & Pronoun Resolution
    4. Photo & Dwell Time Extraction (Profiles with Photos returned even on 0 detections)
    5. Self-Correction SQL Generation & Reflection Loop
    6. Executive-grade Markdown & Entity Synthesis with Data-Driven Follow-ups
    7. Multi-Format UI Dispatcher (Employee Roster, Person Dossier, Camera Traffic, Object Inventory)
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    async def run_query(
        self,
        db: AsyncSession,
        question: str,
        tenant_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint: analyzes user question, resolves context and fuzzy aliases, and returns synthesized answers.
        """
        clean_question = question.strip()
        current_model = model_override or self.client.model

        # 1. Intent Detection: Check if conversational, greeting, or system capability question
        if self._is_conversational_or_help_intent(clean_question):
            direct_answer = await self._handle_conversational_intent(clean_question, current_model)
            return {
                "answer": direct_answer,
                "response_format": "general",
                "sql_query": None,
                "raw_results": [],
                "execution_time_ms": 0.0,
                "entities": [],
                "model_used": current_model,
                "retry_count": 0,
                "suggested_followups": [
                    "List down all employees registered in the directory",
                    "Who stayed the longest on premises?",
                    "Which camera had the highest foot traffic this week?",
                    "List anyone detected carrying a backpack or laptop"
                ],
            }

        # 2. Contextual Query Rewriting for Multi-Turn Dialogue
        enhanced_question = self._resolve_conversational_context(clean_question, conversation_history)

        # 3. Fuzzy Name & Alias Resolution over Database Records (Only for single-person queries)
        fuzzy_match = await self._fuzzy_match_enrolled_person(db, enhanced_question, tenant_id)
        alias_hint = ""
        if fuzzy_match:
            alias_hint = (
                f"\n[Domain Context Note]: The user query refers to enrolled staff member '{fuzzy_match['name']}' "
                f"(Employee Code: '{fuzzy_match.get('employee_code', '')}', Photo: '{fuzzy_match.get('photo_path', '')}'). "
                f"Ensure you query 'employees' and LEFT JOIN 'person_timeline_events' to include their registered photo and dwell time."
            )

        # 4. Build Schema System Prompt & Messages
        system_prompt = build_system_prompt(tenant_id)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Append relevant prior conversation history
        if conversation_history:
            for msg in conversation_history[-4:]:
                content = msg.get("content", "")
                if content:
                    messages.append({"role": msg.get("role", "user"), "content": content})
                
        user_prompt = f"User Question: {enhanced_question}{alias_hint}\nGenerate a single valid PostgreSQL SELECT query for tenant_id = '{tenant_id}'."
        messages.append({"role": "user", "content": user_prompt})

        sql_query = ""
        raw_results: List[Dict[str, Any]] = []
        execution_time_ms: float = 0.0
        used_fallback = False
        retry_count = 0
        last_error = ""

        # 5. SQL Generation via LLM with Fallback Guard
        try:
            raw_response = await self.client.chat(messages=messages, model=current_model)
            sql_query = self.client.extract_sql_from_text(raw_response)
            if not sql_query or not sql_query.lower().startswith("select"):
                sql_query = generate_fallback_sql(enhanced_question, tenant_id)
                used_fallback = True
        except Exception as e:
            logger.warning(f"LLM chat generation failed ({e}). Using intelligent fallback SQL.")
            sql_query = generate_fallback_sql(enhanced_question, tenant_id)
            used_fallback = True

        # 6. Execution & Reflection Loop
        for attempt in range(settings.SQL_AGENT_MAX_RETRIES):
            try:
                raw_results, execution_time_ms = await execute_query_safe(db, sql_query, tenant_id)
                last_error = ""
                break
            except SQLExecutorError as err:
                last_error = str(err)
                logger.warning(f"SQL execution attempt {attempt + 1} failed: {last_error}")
                if used_fallback or attempt == settings.SQL_AGENT_MAX_RETRIES - 1:
                    sql_query = generate_fallback_sql(enhanced_question, tenant_id)
                    try:
                        raw_results, execution_time_ms = await execute_query_safe(db, sql_query, tenant_id)
                        last_error = ""
                        break
                    except Exception as fallback_exc:
                        last_error = str(fallback_exc)
                        break

                # Ask LLM to self-correct the SQL
                messages.append({"role": "assistant", "content": f"```sql\n{sql_query}\n```"})
                messages.append({
                    "role": "user",
                    "content": f"The query failed with error: {last_error}. Please provide the corrected valid PostgreSQL query only."
                })
                try:
                    corrected_resp = await self.client.chat(messages=messages, model=current_model)
                    sql_query = self.client.extract_sql_from_text(corrected_resp)
                    retry_count += 1
                except Exception:
                    sql_query = generate_fallback_sql(enhanced_question, tenant_id)
                    used_fallback = True

        # 7. Fallback to direct fuzzy match if SQL returned 0 rows but a specific person was identified
        if not raw_results and fuzzy_match and not self._is_list_or_aggregate_query(enhanced_question):
            raw_results = [{
                "person_id": fuzzy_match.get("id"),
                "name": fuzzy_match.get("name"),
                "employee_code": fuzzy_match.get("employee_code"),
                "photo_path": fuzzy_match.get("photo_path"),
                "person_type": "employee",
                "total_dwell_seconds": 0,
                "camera_stops_count": 0,
                "cameras_visited": [],
                "note": "Registered in Staff Directory (0 active CCTV detections recorded in queried footage)"
            }]

        # 8. Normalize Row Data (Sanitize Visitor and Staff Names, Format Dwell Times & Attach Photos)
        normalized_results = self._normalize_rows(raw_results)

        # 9. Extract Entities (People with Photos, Cameras, Zones)
        entities = self._extract_entities_from_rows(normalized_results)

        # 10. Determine Specialized Response Format for UI
        response_format = self._determine_response_format(enhanced_question, normalized_results, sql_query)

        # 11. Synthesize Executive Markdown Answer
        answer = await self._synthesize_answer(
            question=enhanced_question,
            sql_query=sql_query,
            rows=normalized_results,
            response_format=response_format,
            error=last_error if not normalized_results and last_error else None,
            model=current_model if not used_fallback else "Deterministic Heuristics Engine",
        )

        suggested_followups = self._generate_followups(enhanced_question, normalized_results)

        return {
            "answer": answer,
            "response_format": response_format,
            "sql_query": sql_query,
            "raw_results": normalized_results,
            "execution_time_ms": execution_time_ms,
            "entities": entities,
            "model_used": current_model if not used_fallback else f"{current_model} (Rule Fallback)",
            "retry_count": retry_count,
            "suggested_followups": suggested_followups,
        }

    @staticmethod
    def _is_list_or_aggregate_query(question: str) -> bool:
        """Checks whether the question is asking for an aggregate/list rather than a single person."""
        q = question.lower().strip()
        list_phrases = [
            "list down", "list of", "list all", "show all", "all employees", "all staff",
            "who are the", "directory", "roster", "everyone", "every employee", "all workers",
            "who visited", "all visitors", "count", "traffic", "hotspot", "ranking", "how many"
        ]
        return any(lp in q for lp in list_phrases)

    @staticmethod
    def _determine_response_format(question: str, rows: List[Dict[str, Any]], sql_query: Optional[str]) -> str:
        """Determines the specialized UI presentation format based on query intent and data structure."""
        q = question.lower().strip()
        if not rows:
            return "general"

        first_row = rows[0]

        # 1. Employee Roster / Directory
        is_emp_query = any(w in q for w in ["employees", "employee list", "all employees", "list employees", "staff directory", "all staff", "who works", "listing down the employees", "list staff"])
        has_many_employees = len(rows) > 1 and all(r.get("person_type") == "employee" or r.get("employee_code") for r in rows)
        if is_emp_query or has_many_employees:
            return "employee_roster"

        # 2. Single Person Dossier
        if len(rows) == 1 and ("name" in first_row or "person_id" in first_row):
            return "person_dossier"
        if any(w in q for w in ["where is", "details for", "dwell of", "dwell time of", "who is", "time spent by"]) and ("name" in first_row or "person_id" in first_row):
            return "person_dossier"

        # 3. Camera Traffic Ranking
        if "camera_name" in first_row and ("unique_visitors_count" in first_row or "total_events_count" in first_row or "avg_dwell_seconds" in first_row):
            return "camera_traffic"
        if any(w in q for w in ["camera", "traffic", "busy", "busiest", "hotspot", "foot traffic", "busiest area"]):
            return "camera_traffic"

        # 4. Object Detection / Belongings
        if any(r.get("associated_objects") for r in rows) or any(w in q for w in ["backpack", "laptop", "bag", "phone", "object", "carrying", "suitcase"]):
            return "object_inventory"

        # 5. Journey / Path Timeline
        if any(w in q for w in ["journey", "path", "trajectory", "timeline", "transit", "steps", "walkway"]):
            return "journey_path"

        # 6. Visitor Roster
        if all(r.get("person_type") == "visitor" or not r.get("employee_code") for r in rows) and len(rows) > 1:
            return "visitor_roster"

        return "general"

    @staticmethod
    async def _fuzzy_match_enrolled_person(db: Any, question: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Fuzzy matches tokens in user question against enrolled staff and visitors in database.
        Strictly excludes general list queries and common English / domain stop words.
        """
        # Guard: Never fuzzy match a single person if query asks for a general list or aggregate!
        if SQLAgent._is_list_or_aggregate_query(question):
            return None

        try:
            if not hasattr(db, "execute"):
                return None

            res = await db.execute(
                text("SELECT id, first_name, last_name, employee_code, photo_path FROM employees WHERE tenant_id = :tid"),
                {"tid": tenant_id}
            )
            emp_rows = res.fetchall()
            
            question_lower = question.lower()
            
            # Domain and grammar stop words to ignore
            stop_words = {
                "the", "and", "for", "with", "from", "where", "how", "what", "which", "when",
                "why", "who", "all", "down", "list", "listing", "show", "showing", "get", "find",
                "tell", "details", "info", "give", "employee", "employees", "staff", "visitor",
                "visitors", "person", "persons", "people", "worker", "workers", "today", "yesterday",
                "time", "dwell", "attendance", "registered", "enrolled", "directory", "roster",
                "any", "are", "is", "was", "were", "did", "does", "me", "his", "her", "their"
            }
            
            tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", question_lower) if len(t) >= 3 and t not in stop_words]
            if not tokens:
                return None

            best_match = None
            best_score = 0.0

            for row in emp_rows:
                emp_id, f_name, l_name, code, photo = str(row[0]), str(row[1] or ""), str(row[2] or ""), str(row[3] or ""), str(row[4] or "")
                full_name = f"{f_name} {l_name}".strip()

                candidates = [c for c in [full_name.lower(), f_name.lower(), l_name.lower(), code.lower()] if len(c) >= 3 and c not in stop_words]

                for cand in candidates:
                    # Exact whole-word match
                    if re.search(rf"\b{re.escape(cand)}\b", question_lower):
                        return {
                            "id": emp_id,
                            "name": full_name,
                            "first_name": f_name,
                            "last_name": l_name,
                            "employee_code": code,
                            "photo_path": photo,
                        }

                    # Fuzzy similarity match across question tokens
                    for token in tokens:
                        score = difflib.SequenceMatcher(None, token, cand).ratio()
                        if score >= 0.85 and score > best_score:
                            best_score = score
                            best_match = {
                                "id": emp_id,
                                "name": full_name,
                                "first_name": f_name,
                                "last_name": l_name,
                                "employee_code": code,
                                "photo_path": photo,
                            }

            if best_score >= 0.85:
                return best_match

        except Exception as e:
            logger.debug(f"Fuzzy name lookup error: {e}")

        return None

    @staticmethod
    def _is_conversational_or_help_intent(text: str) -> bool:
        """Determines if a prompt is conversational, greeting, or system guidance."""
        t = text.lower().strip()
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "sup", "greetings"]
        if t in greetings or any(t.startswith(f"{g} ") for g in ["hi", "hello", "hey"]):
            return True
        
        help_phrases = [
            "what can you do", "who are you", "what is this", "how does this work",
            "help me", "help", "what queries can i ask", "features", "capabilities"
        ]
        if any(hp in t for hp in help_phrases):
            return True

        if t in ["thank you", "thanks", "ok", "cool", "great", "nice", "got it"]:
            return True

        return False

    async def _handle_conversational_intent(self, question: str, model: str) -> str:
        """Generates a professional, context-rich conversational response."""
        prompt = f"""You are the Vigilens Security & Spatio-Temporal Video Intelligence AI Assistant.
The user sent a conversational message: "{question}"

Respond warmly, concisely, and professionally.
Explain that you analyze CCTV surveillance data, tracking individuals, dwell times, employee registered profiles & photos, camera walkways, staff attendance, and detected belongings.
List 3-4 concrete example queries they can ask right away."""

        try:
            resp = await self.client.chat([
                {"role": "system", "content": "You are a professional CCTV security intelligence assistant."},
                {"role": "user", "content": prompt}
            ], model=model, temperature=0.3)
            if resp and len(resp.strip()) > 15:
                return resp.strip()
        except Exception:
            pass

        return (
            "👋 **Hello! I am your Vigilens Security & Video Intelligence Assistant.**\n\n"
            "I connect directly to your CCTV analytics database and employee directory to help you query surveillance data in plain English.\n\n"
            "**Here are things you can ask me:**\n"
            "- 👤 **Staff & Visitors:** *\"Show details for employee Kinjal\"* or *\"Who stayed the longest today?\"*\n"
            "- 📷 **Foot Traffic & Hotspots:** *\"Which camera had the highest foot traffic this week?\"*\n"
            "- 🗺️ **Journey & Paths:** *\"Show the trajectory and camera transit for Visitor #1042\"*\n"
            "- 🎒 **Object Detection:** *\"Which individuals were detected carrying a backpack or laptop?\"*\n"
            "- 📈 **Occupancy Peaks:** *\"What was the peak occupancy recorded across all sessions?\"*\n\n"
            "How can I assist your surveillance monitoring today?"
        )

    @staticmethod
    def _resolve_conversational_context(question: str, history: Optional[List[Dict[str, str]]]) -> str:
        """Resolves pronouns ('he', 'she', 'they', 'the visitor') using prior turns."""
        if not history or len(history) < 2:
            return question

        q = question.lower()
        pronouns = ["he", "she", "they", "this person", "that person", "this visitor", "that visitor", "that camera", "them"]
        needs_resolution = any(re.search(rf"\b{re.escape(p)}\b", q) for p in pronouns)

        if not needs_resolution:
            return question

        last_assistant_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant" or msg.get("sender") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break

        if not last_assistant_msg:
            return question

        name_match = re.search(r"👤 \*\*([^*]+)\*\*", last_assistant_msg) or re.search(r"Visitor #([a-f0-9]+|\d+)", last_assistant_msg)
        if name_match:
            referenced = name_match.group(1).strip()
            return f"{question} (referring to {referenced})"

        cam_match = re.search(r"📷 \*\*([^*]+)\*\*", last_assistant_msg)
        if cam_match:
            cam_name = cam_match.group(1).strip()
            return f"{question} (for camera '{cam_name}')"

        return question

    @staticmethod
    def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans and ensures every visitor or employee name, photo, and dwell formatted string is non-empty."""
        normalized = []
        for r in rows:
            row_dict = dict(r)
            p_id = str(row_dict.get("person_id") or row_dict.get("id") or row_dict.get("identity_id") or "")
            raw_name = str(row_dict.get("name") or row_dict.get("visitor_name") or row_dict.get("employee_name") or "").strip()

            is_emp = bool(row_dict.get("is_employee") or row_dict.get("person_type") == "employee" or row_dict.get("employee_code"))

            if not raw_name or raw_name.lower() in ["unknown", "none", "null", ""]:
                if is_emp:
                    row_dict["name"] = f"Staff Member #{p_id[:6]}" if p_id else "Staff Member"
                else:
                    row_dict["name"] = f"Visitor #{p_id[:6]}" if p_id else "Visitor"
            else:
                row_dict["name"] = raw_name

            # Ensure photo_path is extracted
            photo = row_dict.get("photo_path") or row_dict.get("entry_crop_path") or row_dict.get("exit_crop_path")
            row_dict["photo_path"] = photo if photo and str(photo).lower() != "none" else None

            # Ensure total dwell seconds is numeric and format it
            dwell = row_dict.get("total_dwell_seconds", row_dict.get("dwell_seconds", 0))
            dwell_int = int(dwell) if dwell is not None else 0
            row_dict["total_dwell_seconds"] = dwell_int
            row_dict["dwell_formatted"] = format_dwell_time(dwell_int)

            normalized.append(row_dict)
        return normalized

    async def _synthesize_answer(
        self,
        question: str,
        sql_query: str,
        rows: List[Dict[str, Any]],
        response_format: str = "general",
        error: Optional[str] = None,
        model: str = "gemini-flash-latest",
    ) -> str:
        """Synthesizes raw SQL results into an executive-grade markdown response."""
        if error:
            return f"⚠️ **Could not process surveillance query**\n\n*Error details:* `{error}`\n\nPlease ensure video analytics processing has completed or try rephrasing your question."

        if not rows:
            return (
                "🔍 **No records found matching your query**\n\n"
                "**Diagnostic Breakdown:**\n"
                "- The requested query returned 0 matching records in the database.\n"
                "- If searching for an employee, verify their enrollment under the Staff Directory.\n\n"
                "💡 *Suggestions:*\n"
                "- Try querying staff list: *\"List down all employees registered in the directory\"*\n"
                "- Ask about overall footage: *\"Who visited across all recordings?\"*"
            )

        # Attempt synthesis via LLM
        synthesis_prompt = f"""You are the Vigilens Security & Video Intelligence AI Assistant.
User Question: "{question}"
Response Layout Target: {response_format}
Database Records ({len(rows)} rows found):
{json.dumps(rows[:25], indent=2)}

Instructions:
1. Provide a direct, professional, concise summary answering the user's question.
2. If listing multiple employees ({response_format} == 'employee_roster'), present all {len(rows)} staff members clearly with their names, employee codes, and dwell times.
3. If reporting on a single person ({response_format} == 'person_dossier'), highlight their total accounted dwell time ({rows[0].get('dwell_formatted')}) and camera stops.
4. ALWAYS use the exact non-empty names provided in the records (e.g. "Visitor #a1b2c3" or "Kinjal Patel"). NEVER output empty bold text like "** **" or "**".
5. Keep facts strictly grounded in the provided database records."""

        try:
            synthesis_messages = [
                {"role": "system", "content": "You are a precise, professional surveillance intelligence analyst."},
                {"role": "user", "content": synthesis_prompt}
            ]
            answer = await self.client.chat(messages=synthesis_messages, model=model, temperature=0.2)
            if answer and len(answer.strip()) > 10:
                return answer.strip()
        except Exception:
            pass

        return self._local_template_synthesis(question, rows, response_format)

    @staticmethod
    def _local_template_synthesis(question: str, rows: List[Dict[str, Any]], response_format: str = "general") -> str:
        """Deterministic markdown formatter ensuring clean layouts for employee rosters, single person dossiers, and camera rankings."""
        count = len(rows)
        first_row = rows[0]
        
        # 1. Employee Roster Layout
        if response_format == "employee_roster":
            lines = [f"👥 **Staff Directory ({count} Enrolled Members)**\n"]
            for r in rows:
                name = r.get("name", "Staff Member")
                code = r.get("employee_code", "EMP")
                dwell_str = r.get("dwell_formatted", "0s")
                stops = r.get("camera_stops_count", 0)
                photo_badge = "📸 " if r.get("photo_path") else ""
                lines.append(f"- {photo_badge}**{name}** (`{code}`): Dwell time **{dwell_str}** across **{stops} zone(s)**")
            return "\n".join(lines)

        # 2. Person Dossier Layout
        if response_format == "person_dossier":
            name = first_row.get("name", "Individual")
            ptype = f"STAFF • {first_row.get('employee_code')}" if first_row.get("employee_code") else str(first_row.get("person_type", "individual")).upper()
            dwell_str = first_row.get("dwell_formatted", "0s")
            dwell_secs = first_row.get("total_dwell_seconds", 0)
            stops = first_row.get("camera_stops_count", 0)
            cams = first_row.get("cameras_visited", [])
            cams_str = f", ".join(cams) if isinstance(cams, list) and cams else "None recorded"
            photo_str = "📸 Profile Photo Registered" if first_row.get("photo_path") else "No photo on file"

            lines = [
                f"👤 **Person Profile: {name}** (`{ptype}`)",
                f"- **Status / Photo**: {photo_str}",
                f"- **Total Accounted Dwell Time**: **{dwell_str}** ({dwell_secs}s)",
                f"- **Camera Stops Count**: **{stops}** zone(s)",
                f"- **Locations Visited**: {cams_str}",
            ]
            if first_row.get("first_seen"):
                lines.append(f"- **First Seen**: `{first_row.get('first_seen')}`")
            if first_row.get("last_seen"):
                lines.append(f"- **Last Seen**: `{first_row.get('last_seen')}`")
            return "\n".join(lines)

        # 3. Camera Traffic Layout
        if response_format == "camera_traffic" or "camera_name" in first_row:
            lines = [f"📷 **Camera Foot Traffic Analysis ({count} cameras monitored)**\n"]
            for r in rows:
                cam = r.get("camera_name", "Camera")
                visitors = r.get("unique_visitors_count", r.get("total_events_count", 0))
                avg_dwell = format_dwell_time(int(r.get("avg_dwell_seconds", 0)))
                lines.append(f"- 📹 **{cam}**: **{visitors} unique visitor(s)** (Avg dwell: **{avg_dwell}**)")
            return "\n".join(lines)

        # 4. General / People Fallback Layout
        lines = [f"Found **{count} record(s)** for your query:\n"]
        if "name" in first_row or "person_id" in first_row:
            for r in rows[:15]:
                p_id = str(r.get("person_id") or r.get("id") or "")
                name = str(r.get("name") or "").strip()
                if not name or name.lower() in ["unknown", "none", "null"]:
                    name = f"Visitor #{p_id[:6]}" if p_id else "Visitor"

                emp_code = r.get("employee_code")
                ptype = f"STAFF • {emp_code}" if emp_code else str(r.get("person_type", "visitor")).upper()
                dwell_str = r.get("dwell_formatted", f"{r.get('total_dwell_seconds', 0)}s")
                stops = r.get("camera_stops_count", r.get("camera_count", 0))
                photo_badge = " 📸 *[Photo]*" if r.get("photo_path") else ""
                lines.append(f"- 👤 **{name}** (`{ptype}`){photo_badge}: Dwell **{dwell_str}** in **{stops} zone(s)**")
        else:
            keys = list(first_row.keys())[:6]
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for r in rows[:10]:
                row_vals = [str(r.get(k, "")) for k in keys]
                lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

    @staticmethod
    def _extract_entities_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts structured person (with photos & dwell times), camera, and zone entity objects."""
        entities: List[Dict[str, Any]] = []
        seen = set()

        for r in rows:
            # Person entity
            p_id = str(r.get("person_id") or r.get("identity_id") or r.get("employee_id") or r.get("id") or "")
            raw_name = str(r.get("name") or r.get("visitor_name") or r.get("employee_name") or "").strip()
            
            if p_id and p_id not in seen:
                seen.add(p_id)
                clean_label = raw_name if raw_name and raw_name.lower() not in ["unknown", "none", "null"] else f"Visitor #{p_id[:6]}"
                photo = r.get("photo_path")
                clean_photo_path = str(photo).replace("\\", "/").lstrip("/") if photo else ""
                photo_url = f"http://localhost:8000/{clean_photo_path}" if photo else None
                cameras = r.get("cameras_visited", [])
                if not isinstance(cameras, list):
                    cameras = [cameras] if cameras else []
                
                dwell_sec = int(r.get("total_dwell_seconds", r.get("dwell_seconds", 0)))
                entities.append({
                    "type": "person",
                    "id": p_id,
                    "label": clean_label,
                    "photo_path": photo,
                    "photo_url": photo_url,
                    "dwell_seconds": dwell_sec,
                    "dwell_formatted": format_dwell_time(dwell_sec),
                    "person_type": r.get("person_type") or ("employee" if r.get("employee_code") else "visitor"),
                    "employee_code": r.get("employee_code"),
                    "cameras_visited": cameras,
                })

            # Camera entity
            cam = r.get("camera_name")
            if cam and cam not in seen:
                seen.add(cam)
                entities.append({
                    "type": "camera",
                    "label": cam,
                    "photo_path": None,
                })

            # Zone entity
            zone = r.get("zone_name")
            if zone and zone not in seen:
                seen.add(zone)
                entities.append({
                    "type": "zone",
                    "label": zone,
                    "photo_path": None,
                })

        return entities[:20]

    @staticmethod
    def _generate_followups(question: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Generates dynamic, data-aware follow-up suggestion queries."""
        q = question.lower()
        
        if rows and ("name" in rows[0] or "person_id" in rows[0]):
            top_person = rows[0]
            top_name = top_person.get("name", "top person")
            target = top_name if top_name and not top_name.startswith("Visitor #") else f"person {str(top_person.get('person_id') or top_person.get('id', ''))[:6]}"
            return [
                f"Show complete journey and camera timeline for {target}",
                "Which camera had the highest foot traffic?",
                "Show all staff members detected today"
            ]

        if "camera" in q or "traffic" in q:
            return [
                "Who stayed the longest on premises?",
                "Show all line crossing entry events",
                "List all new visitors detected this week"
            ]

        return [
            "Who stayed the longest on premises?",
            "Which camera had the highest foot traffic?",
            "Show me all staff members detected today",
            "Show anyone detected with a backpack or laptop"
        ]
