import uuid
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database.session import get_db
from shared.dependencies.auth import get_current_user
from modules.users.model import User
from application.sql_agent.schema import (
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantQueryResponseData,
    ModelHealthResponse,
    ModelInfo,
)
from application.sql_agent.agent import SQLAgent
from application.sql_agent.llm_client import OllamaClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["Vigilens AI Assistant (SQL Agent)"])
agent = SQLAgent()

async def get_optional_user_or_tenant(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[uuid.UUID]:
    """Extracts tenant_id from auth credentials or defaults to first active tenant."""
    try:
        user = await get_current_user(request, db)
        if user and user.tenant_id:
            return user.tenant_id
    except Exception:
        pass
        
    # Query the first available tenant in DB if unauthenticated (demo/standalone dev mode)
    try:
        res = await db.execute(text("SELECT tenant_id FROM camera_nodes LIMIT 1"))
        row = res.fetchone()
        if row and row[0]:
            return row[0]
            
        res_ident = await db.execute(text("SELECT tenant_id FROM advanced_person_identities LIMIT 1"))
        row_ident = res_ident.fetchone()
        if row_ident and row_ident[0]:
            return row_ident[0]
    except Exception:
        pass

    return uuid.UUID("00000000-0000-0000-0000-000000000000")


@router.post(
    "/query",
    response_model=AssistantQueryResponse,
    summary="Query Vigilens surveillance data with Natural Language SQL Agent",
)
async def query_assistant(
    payload: AssistantQueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Translates natural language questions into safe PostgreSQL queries via Ollama,
    executes them against the spatio-temporal database, and returns formatted markdown with entity links.
    """
    tenant_id: Optional[uuid.UUID] = None
    
    # Priority 1: explicitly passed workspace_id in payload
    if payload.workspace_id:
        try:
            tenant_id = uuid.UUID(payload.workspace_id)
        except ValueError:
            pass

    # Priority 2: Authenticated user context or DB tenant discovery
    if not tenant_id:
        tenant_id = await get_optional_user_or_tenant(request, db)

    try:
        result = await agent.run_query(
            db=db,
            question=payload.question,
            tenant_id=str(tenant_id),
            conversation_history=payload.conversation_history,
            model_override=payload.model_name,
        )

        response_data = AssistantQueryResponseData(
            answer=result["answer"],
            response_format=result.get("response_format", "general"),
            sql_query=result.get("sql_query"),
            raw_results=result.get("raw_results"),
            execution_time_ms=result.get("execution_time_ms"),
            model_used=result.get("model_used"),
            retry_count=result.get("retry_count", 0),
            entities=result.get("entities"),
            suggested_followups=result.get("suggested_followups"),
            session_id=payload.session_id or f"session_{uuid.uuid4().hex[:8]}",
        )

        return AssistantQueryResponse(
            status=status.HTTP_200_OK,
            message="Query processed successfully",
            data=response_data,
        )
    except Exception as exc:
        logger.exception("Error executing SQL Agent query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(exc)}",
        )


@router.get(
    "/models",
    response_model=ModelHealthResponse,
    summary="Check Ollama connection health and list installed models",
)
async def get_ollama_models():
    """Returns Ollama status and installed models."""
    health = await agent.client.check_health()
    return ModelHealthResponse(
        status=status.HTTP_200_OK,
        message="LLM status retrieved",
        data=ModelInfo(
            status=health["status"],
            provider=health.get("provider", "Cloud"),
            base_url=health.get("base_url"),
            current_model=health["current_model"],
            available_models=health.get("available_models", []),
            error=health.get("error"),
        ),
    )


@router.get(
    "/schema-overview",
    summary="Returns available surveillance tables and query metrics",
)
async def get_schema_overview():
    """Returns overview of database tables accessible to the SQL agent."""
    return {
        "status": 200,
        "message": "Schema overview",
        "data": {
            "tables": [
                {
                    "name": "camera_nodes",
                    "description": "Physical CCTV Camera locations and floor plan nodes",
                },
                {
                    "name": "camera_zones",
                    "description": "Spatial zones (entry, exit, crossing) inside camera fields of view",
                },
                {
                    "name": "employees",
                    "description": "Enrolled staff members with photos and employee codes",
                },
                {
                    "name": "advanced_person_identities",
                    "description": "Detected individuals across all camera runs with ReID & face tags",
                },
                {
                    "name": "person_timeline_events",
                    "description": "Complete spatiotemporal journey events with entry/exit timestamps and dwell duration",
                },
                {
                    "name": "advanced_people_analytics_sessions",
                    "description": "Footage analysis batches with peak occupancy, visitor counts, and timeline JSONs",
                },
                {
                    "name": "advanced_visitor_attendance_logs",
                    "description": "Daily visitor attendance records",
                },
                {
                    "name": "advanced_employee_attendance_logs",
                    "description": "Daily employee attendance records",
                },
            ]
        },
    }
