from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.tools import ToolRegistry

router = APIRouter(prefix="/api")
tool_registry = ToolRegistry()


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: dict


class ToolExecutionResponse(BaseModel):
    html: str


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(request: ToolExecutionRequest):
    """Execute a tool and return the result HTML"""
    html = tool_registry.execute_tool(request.tool_name, request.arguments)
    return ToolExecutionResponse(html=html)
