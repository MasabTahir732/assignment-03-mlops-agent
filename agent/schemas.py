"""Validated inputs and evidence envelopes shared by every interface."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(ToolInput):
    pass


class DvcCompareInput(ToolInput):
    old: str = Field(pattern=r"^[A-Za-z0-9_./-]+$", max_length=100)
    new: str = Field(default="HEAD", pattern=r"^[A-Za-z0-9_./-]+$", max_length=100)


class ReproduceInput(ToolInput):
    revision: str = Field(pattern=r"^[A-Za-z0-9_./-]+$", max_length=100)


class RunsInput(ToolInput):
    limit: int = Field(default=5, ge=1, le=50)


class BestInput(RunsInput):
    metric: Literal["f1", "accuracy"] = "f1"
    mode: Literal["max", "min"] = "max"


class RunInput(ToolInput):
    run_id: str = Field(pattern=r"^[a-fA-F0-9]{32}$")


class CompareRunsInput(ToolInput):
    run_ids: list[str] = Field(min_length=2, max_length=10)


class DagInput(ToolInput):
    dag_id: Literal["iris_training"] = "iris_training"


class TriggerInput(DagInput):
    confirmed: bool = False
    inject_failure: Literal["none", "validation", "quality"] = "none"


class TaskInput(DagInput):
    dag_run_id: str = Field(min_length=1, max_length=200)


class PodInput(ToolInput):
    pod: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$", max_length=253)
    lines: int = Field(default=80, ge=1, le=200)


class RestartInput(ToolInput):
    namespace: str = Field(default="mlops-demo", pattern=r"^[a-z0-9-]+$")
    confirmed: bool = False


class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "error"]
    data: Any = None
    error: str | None = None
    request_id: str
    duration_ms: float


class AgentResponse(BaseModel):
    request_id: str
    mode: str
    facts: list[str]
    uncertainties: list[str]
    recommendations: list[str]
    evidence: list[ToolResult]
