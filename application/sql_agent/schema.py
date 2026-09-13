from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class EntityItem(BaseModel):
    type: str = Field(description="Entity type: 'person' | 'camera' | 'zone' | 'date'")
    id: Optional[str] = Field(default=None, description="UUID or ID of the entity if applicable")
    label: str = Field(description="Display label or name of the entity")

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    workspace_id: Optional[str] = Field(default=None, description="Active workspace/tenant ID")
    session_id: Optional[str] = Field(default=None, description="Chat session ID for context retention")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior conversation messages")
    model_name: Optional[str] = Field(default=None, description="Ollama model override")

class AssistantQueryResponseData(BaseModel):
    answer: str
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
