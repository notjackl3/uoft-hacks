"""
FastAPI Routes for Company Documentation Management

Endpoints for creating companies, ingesting docs, and querying procedures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, HttpUrl

from app.services.graph import graph_service
from app.services.doc_ingestion import ingest_docs_from_url, ingest_docs_from_file
from app.services.orchestration import plan_next_ui_action, get_relevant_context

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateCompanyRequest(BaseModel):
    name: str
    domain: Optional[str] = None


class CreateCompanyResponse(BaseModel):
    id: str
    name: str
    domain: Optional[str]


class IngestUrlRequest(BaseModel):
    root_url: str
    max_pages: int = 50


class IngestResponse(BaseModel):
    source_id: str
    pages_crawled: int
    chunks_created: int
    procedures_extracted: int
    status: str


class ProcedureResponse(BaseModel):
    id: str
    goal: str
    source_url: Optional[str]
    source_title: Optional[str]
    steps: List[Dict[str, Any]]


class PlanActionRequest(BaseModel):
    user_goal: str
    ui_context: Dict[str, Any]
    session_state: Optional[Dict[str, Any]] = None


class PlanActionResponse(BaseModel):
    procedure_id: Optional[str]
    step_id: Optional[str]
    action: Dict[str, Any]
    confidence: float
    justification: str


# ============================================================================
# Company Endpoints
# ============================================================================

@router.post("", response_model=CreateCompanyResponse, status_code=status.HTTP_201_CREATED)
async def create_company(request: CreateCompanyRequest):
    """Create a new company for documentation management."""
    try:
        company = graph_service.create_company(
            name=request.name,
            domain=request.domain
        )
        if not company:
            raise HTTPException(status_code=500, detail="Failed to create company")
        
        return CreateCompanyResponse(
            id=company["id"],
            name=company["name"],
            domain=company.get("domain")
        )
    except Exception as e:
        logger.exception(f"Error creating company: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Dict[str, Any]])
async def list_companies():
    """List all companies."""
    try:
        companies = graph_service.list_companies()
        return companies
    except Exception as e:
        logger.exception(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{company_id}")
async def get_company(company_id: str):
    """Get a company by ID."""
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


# ============================================================================
# Document Ingestion Endpoints
# ============================================================================

@router.post("/{company_id}/docs/url", response_model=IngestResponse)
async def ingest_docs_url(company_id: str, request: IngestUrlRequest):
    """
    Ingest documentation from a root URL.
    
    Crawls the site, extracts content, chunks, embeds, and extracts procedures.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        result = await ingest_docs_from_url(
            company_id=company_id,
            root_url=request.root_url,
            max_pages=request.max_pages
        )
        return IngestResponse(**result)
    except Exception as e:
        logger.exception(f"Error ingesting docs from URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{company_id}/docs/upload", response_model=IngestResponse)
async def ingest_docs_upload(
    company_id: str,
    file: UploadFile = File(...)
):
    """
    Ingest documentation from an uploaded file.
    
    Supports: text/plain, text/markdown, text/html, application/pdf
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Validate file type
    allowed_types = [
        "text/plain",
        "text/markdown",
        "text/html",
        "application/xhtml+xml",
        "application/pdf",
        "application/octet-stream"  # For unknown types, we'll try anyway
    ]
    
    content_type = file.content_type or "application/octet-stream"
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}"
        )
    
    try:
        content = await file.read()
        result = await ingest_docs_from_file(
            company_id=company_id,
            filename=file.filename or "uploaded_file",
            content=content,
            content_type=content_type
        )
        return IngestResponse(**result)
    except Exception as e:
        logger.exception(f"Error ingesting uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Procedure Endpoints
# ============================================================================

@router.get("/{company_id}/procedures", response_model=List[ProcedureResponse])
async def list_procedures(company_id: str):
    """
    List all procedures extracted for a company.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        procedures = graph_service.get_procedures_for_company(company_id)
        return [
            ProcedureResponse(
                id=p["id"],
                goal=p["goal"],
                source_url=p.get("source_url"),
                source_title=p.get("source_title"),
                steps=p.get("steps", [])
            )
            for p in procedures
        ]
    except Exception as e:
        logger.exception(f"Error listing procedures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{company_id}/procedures/{procedure_id}")
async def get_procedure(company_id: str, procedure_id: str):
    """
    Get a specific procedure with all its steps.
    """
    procedure = graph_service.get_procedure_with_steps(procedure_id)
    if not procedure:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return procedure


# ============================================================================
# Navigation Planning Endpoints
# ============================================================================

@router.post("/{company_id}/plan-action", response_model=PlanActionResponse)
async def plan_action(company_id: str, request: PlanActionRequest):
    """
    Plan the next UI action based on user goal and current UI state.
    
    Uses the company's ingested documentation to find matching procedures
    and determine the appropriate next step.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        result = await plan_next_ui_action(
            company_id=company_id,
            user_goal=request.user_goal,
            ui_context=request.ui_context,
            session_state=request.session_state
        )
        return PlanActionResponse(**result)
    except Exception as e:
        logger.exception(f"Error planning action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{company_id}/context")
async def get_context(
    company_id: str,
    user_goal: str,
    ui_context: Dict[str, Any],
    limit: int = 5
):
    """
    Get relevant documentation context for a goal and UI state.
    
    Useful for providing context to an LLM for complex reasoning.
    """
    # Verify company exists
    company = graph_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    try:
        chunks = await get_relevant_context(
            company_id=company_id,
            user_goal=user_goal,
            ui_context=ui_context,
            limit=limit
        )
        return {"chunks": chunks}
    except Exception as e:
        logger.exception(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Schema Setup Endpoint (Admin)
# ============================================================================

@router.post("/admin/setup-schema")
async def setup_schema():
    """
    Set up Neo4j schema (constraints and indexes).
    
    Should be called once when setting up the system.
    """
    try:
        graph_service.setup_schema()
        graph_service.setup_vector_index()
        return {"status": "Schema setup complete"}
    except Exception as e:
        logger.exception(f"Error setting up schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))
