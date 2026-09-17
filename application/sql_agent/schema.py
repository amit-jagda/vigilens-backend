from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class EntityItem(BaseModel):
    type: str = Field(description="Entity type: 'person' | 'camera' | 'zone' | 'date'")
    id: Optional[str] = Field(default=None, description="UUID or ID of the entity if applicable")
    label: str = Field(description="Display label or name of the entity")
    photo_path: Optional[str] = Field(default=None, description="Relative path to employee photo or visitor crop")
    photo_url: Optional[str] = Field(default=None, description="Full or resolved streaming URL to employee/visitor photo")
    dwell_seconds: Optional[float] = Field(default=None, description="Total dwell time in seconds")
    dwell_formatted: Optional[str] = Field(default=None, description="Human readable dwell string e.g. '3m 15s'")
    person_type: Optional[str] = Field(default=None, description="'employee' or 'visitor'")
    employee_code: Optional[str] = Field(default=None, description="Employee code e.g. EMP001")
    cameras_visited: Optional[List[str]] = Field(default=None, description="List of cameras visited")

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    workspace_id: Optional[str] = Field(default=None, description="Active workspace/tenant ID")
    session_id: Optional[str] = Field(default=None, description="Chat session ID for context retention")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior conversation messages")
    model_name: Optional[str] = Field(default=None, description="Ollama model override")

class AssistantQueryResponseData(BaseModel):
    answer: str
    response_format: Optional[str] = Field(default="general", description="UI response format: 'employee_roster' | 'person_dossier' | 'visitor_roster' | 'camera_traffic' | 'journey_path' | 'object_inventory' | 'general'")
    sql_query: Optional[str] = None
    raw_results: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: Optional[float] = None
    model_used: Optional[str] = None
    retry_count: Optional[int] = 0
    entities: Optional[List[EntityItem]] = None
    suggested_followups: Optional[List[str]] = None
    session_id: Optional[str] = None

class AssistantQueryResponse(BaseModel):
    status: int = 200
    message: str = "Query processed successfully"
    data: AssistantQueryResponseData

class ModelInfo(BaseModel):
    status: str
    provider: Optional[str] = "Cloud"
    base_url: Optional[str] = None
    current_model: str
    available_models: List[str]
    error: Optional[str] = None

class ModelHealthResponse(BaseModel):
    status: int = 200
    message: str = "LLM Engine status retrieved"
    data: ModelInfo
