"""
Models package - contains all Pydantic models.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PageFeature(BaseModel):
    index: int
    type: Literal["input", "button", "link"]
    text: Optional[str] = ""
    selector: str
    href: Optional[str] = None
    placeholder: Optional[str] = None
    aria_label: Optional[str] = None
    value_len: Optional[int] = 0
    already_clicked: Optional[bool] = False  # Track if element was interacted with before


class TargetHints(BaseModel):
    type: Optional[str] = None
    text_contains: List[str] = Field(default_factory=list)
    placeholder_contains: List[str] = Field(default_factory=list)
    selector_pattern: Optional[str] = None
    role: Optional[str] = None

    @field_validator("text_contains", "placeholder_contains", mode="before")
    @classmethod
    def _coerce_list_fields(cls, v):
        # LLMs sometimes output null for list fields; treat that as empty list.
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None and str(x).strip()]
        if isinstance(v, str):
            return [v] if v.strip() else []
        # Fallback: unknown type -> empty
        return []


class PlannedStep(BaseModel):
    step_number: int
    action: Literal["CLICK", "TYPE", "SCROLL", "WAIT", "DONE"]
    description: str
    target_hints: TargetHints = Field(default_factory=TargetHints)
    text_input: Optional[str] = None
    expected_page_change: bool = False


class StartSessionRequest(BaseModel):
    user_goal: str
    initial_page_features: List[PageFeature]
    url: str
    page_title: str


class StartSessionResponse(BaseModel):
    session_id: str
    planned_steps: List[PlannedStep]
    total_steps: int
    first_step: dict


class PreviousActionResult(BaseModel):
    success: bool = True
    error: Optional[str] = None


class NextActionRequest(BaseModel):
    session_id: str
    page_features: List[PageFeature]
    url: Optional[str] = None
    page_title: Optional[str] = None
    previous_action_result: PreviousActionResult = Field(default_factory=PreviousActionResult)


class NextActionResponse(BaseModel):
    step_number: int
    total_steps: int
    action: str
    target_feature_index: Optional[int]
    target_feature: Optional[PageFeature]
    instruction: str
    text_input: Optional[str] = None
    confidence: float
    expected_page_change: bool
    session_complete: bool = False


class CorrectionRequest(BaseModel):
    session_id: str
    feedback: Literal["wrong_element", "doesnt_work"]
    actual_feature_index: Optional[int] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    status: Literal["in_progress", "completed", "failed"]
    current_step_number: int
    total_steps: int
    user_goal: str
    url: str
    updated_at: datetime


# Export all models
__all__ = [
    "PageFeature",
    "TargetHints",
    "PlannedStep",
    "StartSessionRequest",
    "StartSessionResponse",
    "PreviousActionResult",
    "NextActionRequest",
    "NextActionResponse",
    "CorrectionRequest",
    "SessionStatusResponse",
]
