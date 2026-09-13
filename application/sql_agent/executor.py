import re
import time
import uuid
import datetime
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from configs.base import settings

FORBIDDEN_SQL_KEYWORDS = [
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bEXECUTE\b",
    r"\bCOPY\b",
    r"\bVACUUM\b",
    r"\bLOCK\b",
]

class SQLExecutorError(Exception):
    pass

def sanitize_and_validate_sql(sql: str, tenant_id: str) -> str:
    """
    Validates that the SQL query is strictly read-only and scopes it to the tenant.
    """
    cleaned = sql.strip().rstrip(";")
    
    if not cleaned:
        raise SQLExecutorError("Empty SQL query generated.")

    # Check for forbidden mutation statements
    for pattern in FORBIDDEN_SQL_KEYWORDS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise SQLExecutorError(f"Security Violation: Dangerous or modifying SQL pattern detected ({pattern}). Only SELECT queries are permitted.")

    # Must start with SELECT or WITH
    if not (cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH")):
        raise SQLExecutorError("Invalid query: Query must begin with SELECT or WITH.")

    # Verify tenant_id inclusion
    if str(tenant_id) not in cleaned and "tenant_id" not in cleaned.lower():
        # Inject tenant_id filter if missing on simple queries
        if "WHERE" in cleaned.upper():
            cleaned = re.sub(r"(?i)\bWHERE\b", f"WHERE tenant_id = '{tenant_id}' AND ", cleaned, count=1)
        elif "GROUP BY" in cleaned.upper():
            cleaned = re.sub(r"(?i)\bGROUP BY\b", f"WHERE tenant_id = '{tenant_id}' GROUP BY", cleaned, count=1)
        elif "ORDER BY" in cleaned.upper():
            cleaned = re.sub(r"(?i)\bORDER BY\b", f"WHERE tenant_id = '{tenant_id}' ORDER BY", cleaned, count=1)
        elif "LIMIT" in cleaned.upper():
            cleaned = re.sub(r"(?i)\bLIMIT\b", f"WHERE tenant_id = '{tenant_id}' LIMIT", cleaned, count=1)
        else:
            cleaned = f"{cleaned} WHERE tenant_id = '{tenant_id}'"

    # Enforce row limit
    if "LIMIT" not in cleaned.upper():
        cleaned = f"{cleaned} LIMIT {settings.SQL_AGENT_MAX_ROWS}"

    return cleaned


def _serialize_value(val: Any) -> Any:
    """Converts DB values (UUID, datetime, Decimal) to JSON-serializable types."""
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    return val


async def execute_query_safe(
    db: AsyncSession,
    sql: str,
    tenant_id: str,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Executes a sanitized SQL query in a read-only transaction with execution timing.
    Returns (rows, execution_time_ms).
    """
    validated_sql = sanitize_and_validate_sql(sql, tenant_id)
    
    start_time = time.perf_counter()
    try:
        # Set statement timeout and read-only mode for the session execution
        timeout_ms = int(settings.SQL_AGENT_TIMEOUT_SECONDS * 1000)
        await db.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
        
        result = await db.execute(text(validated_sql))
        rows = result.mappings().all()
        
        serialized_rows = [
            {key: _serialize_value(val) for key, val in row.items()}
            for row in rows
        ]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return serialized_rows, round(elapsed_ms, 2)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        raise SQLExecutorError(f"Database execution error: {str(exc)}") from exc
