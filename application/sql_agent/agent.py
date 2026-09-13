import logging
import json
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from configs.base import settings
from application.sql_agent.schema_context import build_system_prompt
from application.sql_agent.llm_client import OllamaClient, generate_fallback_sql
from application.sql_agent.executor import execute_query_safe, SQLExecutorError

logger = logging.getLogger(__name__)

class SQLAgent:
    """
    Autonomous Spatio-Temporal SQL Agent powered by Ollama.
    Handles Natural Language -> SQL -> Safe PostgreSQL Execution -> Reflection -> Markdown & Entity Synthesis.
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
        Main entrypoint: executes natural language question over tenant database.
        """
        system_prompt = build_system_prompt(tenant_id)
        current_model = model_override or self.client.model
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Append relevant prior conversation context
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                
        user_prompt = f"Question: {question}\nGenerate a single valid PostgreSQL SELECT query for tenant_id = '{tenant_id}'."
        messages.append({"role": "user", "content": user_prompt})

        sql_query = ""
        raw_results: List[Dict[str, Any]] = []
        execution_time_ms: float = 0.0
        used_fallback = False
        retry_count = 0
        last_error = ""

        # 1. SQL Generation with Ollama & Self-Correction Reflection Loop
        try:
            raw_response = await self.client.chat(messages=messages, model=current_model)
            sql_query = self.client.extract_sql_from_text(raw_response)
        except Exception as e:
            logger.warning(f"Ollama chat generation failed ({e}). Using deterministic fallback SQL.")
            sql_query = generate_fallback_sql(question, tenant_id)
            used_fallback = True

        # 2. Execution & Reflection Loop
        for attempt in range(settings.SQL_AGENT_MAX_RETRIES):
            try:
                raw_results, execution_time_ms = await execute_query_safe(db, sql_query, tenant_id)
                last_error = ""
                break
            except SQLExecutorError as err:
                last_error = str(err)
                logger.warning(f"SQL execution attempt {attempt + 1} failed: {last_error}")
                if used_fallback or attempt == settings.SQL_AGENT_MAX_RETRIES - 1:
                    # If fallback also failed or retries exhausted, switch to basic fallback
                    sql_query = generate_fallback_sql(question, tenant_id)
                    try:
                        raw_results, execution_time_ms = await execute_query_safe(db, sql_query, tenant_id)
                        last_error = ""
                        break
                    except Exception as fallback_exc:
                        last_error = str(fallback_exc)
                        break

                # Ask Ollama to reflect and correct the SQL
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
                    sql_query = generate_fallback_sql(question, tenant_id)
                    used_fallback = True

        # 3. Extract Entities (People, Cameras, Zones) from Results
        entities = self._extract_entities_from_rows(raw_results)

        # 4. Synthesize Markdown Answer
        answer = await self._synthesize_answer(
            question=question,
            sql_query=sql_query,
            rows=raw_results,
            error=last_error if not raw_results and last_error else None,
            model=current_model if not used_fallback else "Deterministic Heuristics Engine",
        )

        suggested_followups = self._generate_followups(question, raw_results)

        return {
            "answer": answer,
            "sql_query": sql_query,
            "raw_results": raw_results,
            "execution_time_ms": execution_time_ms,
            "entities": entities,
            "model_used": current_model if not used_fallback else f"{current_model} (Rule Fallback)",
            "retry_count": retry_count,
            "suggested_followups": suggested_followups,
        }

    async def _synthesize_answer(
        self,
        question: str,
        sql_query: str,
        rows: List[Dict[str, Any]],
        error: Optional[str] = None,
        model: str = "ollama",
    ) -> str:
        """
        Synthesizes raw SQL results into a rich markdown response.
        """
        if error:
            return f"⚠️ **Could not process surveillance query**\n\n*Error details:* `{error}`\n\nPlease ensure footage has been analyzed for this workspace or try rephrasing your question."

        if not rows:
            return "🔍 **No records found** matching your query in the current workspace.\n\n*Suggestions:*\n- Check if video footage for the specified date range has completed processing.\n- Try broadening your search or asking about total visits."

        # Attempt synthesis via Ollama for natural conversational output
        synthesis_prompt = f"""You are the Vigilens Security & Video Intelligence AI Assistant.
User Question: "{question}"
Database Records ({len(rows)} rows found):
{json.dumps(rows[:15], indent=2)}

Instructions:
1. Provide a concise, professional summary answering the user's question directly.
2. Highlight key individuals, camera locations, dwell times, and counts using bold text.
3. Use bullet points or markdown tables where helpful.
4. Keep the response factual based ONLY on the provided database records."""

        try:
            synthesis_messages = [
                {"role": "system", "content": "You are a concise surveillance analytics assistant."},
                {"role": "user", "content": synthesis_prompt}
            ]
            answer = await self.client.chat(messages=synthesis_messages, temperature=0.2)
            if answer and len(answer.strip()) > 10:
                return answer.strip()
        except Exception:
            pass

        # Fallback rich template synthesizer if LLM synthesis is skipped
        return self._local_template_synthesis(question, rows)

    @staticmethod
    def _local_template_synthesis(question: str, rows: List[Dict[str, Any]]) -> str:
        """Deterministic markdown formatter for table results."""
        count = len(rows)
        first_row = rows[0]
        
        lines = [f"Found **{count} record(s)** for your query:\n"]
        
        # Check if rows represent people
        if "name" in first_row or "person_id" in first_row:
            for r in rows[:10]:
                name = r.get("name", "Unknown")
                ptype = r.get("person_type", "visitor").upper()
                dwell = r.get("total_dwell_seconds", r.get("dwell_seconds", 0))
                stops = r.get("camera_stops_count", r.get("camera_count", 1))
                cams = r.get("cameras_visited", [])
                cams_str = f" across {', '.join(cams)}" if cams else ""
                lines.append(f"- 👤 **{name}** (`{ptype}`): Dwell time **{int(dwell)}s** in **{stops} zone(s)**{cams_str}")
        elif "camera_name" in first_row:
            for r in rows[:10]:
                cam = r.get("camera_name", "Camera")
                visitors = r.get("unique_visitors_count", r.get("total_events_count", 0))
                avg_dwell = r.get("avg_dwell_seconds", 0)
                lines.append(f"- 📷 **{cam}**: **{visitors} unique visitor(s)** (Avg dwell: **{int(avg_dwell)}s**)")
        else:
            # Generic table representation
            keys = list(first_row.keys())[:5]
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for r in rows[:10]:
                row_vals = [str(r.get(k, "")) for k in keys]
                lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

    @staticmethod
    def _extract_entities_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extracts structured person, camera, and zone entity objects for interactive frontend chips."""
        entities: List[Dict[str, str]] = []
        seen = set()

        for r in rows:
            # Person entity
            p_id = r.get("person_id") or r.get("identity_id") or r.get("employee_id")
            p_name = r.get("name") or r.get("visitor_name") or r.get("employee_name")
            if p_id and str(p_id) not in seen:
                seen.add(str(p_id))
                entities.append({
                    "type": "person",
                    "id": str(p_id),
                    "label": p_name or f"Person #{str(p_id)[:6]}"
                })

            # Camera entity
            cam = r.get("camera_name")
            if cam and cam not in seen:
                seen.add(cam)
                entities.append({
                    "type": "camera",
                    "label": cam
                })

            # Zone entity
            zone = r.get("zone_name")
            if zone and zone not in seen:
                seen.add(zone)
                entities.append({
                    "type": "zone",
                    "label": zone
                })

        return entities[:12]

    @staticmethod
    def _generate_followups(question: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Generates dynamic, context-aware follow-up suggestion queries."""
        q = question.lower()
        if "longest" in q or "dwell" in q:
            return [
                "Which camera has the highest foot traffic?",
                "Show me all staff members detected today",
                "Who visited during the last 24 hours?"
            ]
        if "camera" in q or "traffic" in q:
            return [
                "Who stayed the longest on premises?",
                "List all new visitors detected this week",
                "Show all line crossing entry events"
            ]
        return [
            "Who stayed the longest on premises?",
            "Which camera had the highest foot traffic?",
            "Show me all staff members detected today"
        ]
