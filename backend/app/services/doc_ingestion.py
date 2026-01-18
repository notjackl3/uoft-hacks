"""
Document Ingestion Service

Handles crawling, chunking, embedding, and procedure extraction from documentation.
"""

from __future__ import annotations

import logging
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.services.graph import graph_service

logger = logging.getLogger(__name__)

# ============================================================================
# Crawling
# ============================================================================

def crawl_docs(
    root_url: str,
    max_pages: int = 50,
    same_domain_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Crawl documentation starting from a root URL.
    
    Returns list of {url, title, text, headings}
    """
    parsed_root = urlparse(root_url)
    domain = parsed_root.netloc
    
    visited = set()
    to_visit = [root_url]
    pages = []
    
    logger.info(f"Starting crawl from {root_url} (max {max_pages} pages)")
    
    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        
        # Normalize URL
        url = url.split("#")[0]  # Remove fragment
        if url in visited:
            continue
        
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "DocIngestion/1.0 (Knowledge Base Builder)"
            })
            
            if response.status_code != 200:
                logger.debug(f"Skipping {url} (status {response.status_code})")
                continue
            
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
                tag.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text(strip=True)
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)
            
            # Extract headings
            headings = []
            for h in soup.find_all(["h1", "h2", "h3", "h4"]):
                text = h.get_text(strip=True)
                if text and len(text) < 200:
                    headings.append(text)
            
            # Extract main content
            main = soup.find("main") or soup.find("article") or soup.find(class_=re.compile(r"content|docs|documentation"))
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
            
            # Skip if too little content
            if len(text) < 100:
                continue
            
            pages.append({
                "url": url,
                "title": title,
                "text": text,
                "headings": headings
            })
            
            logger.info(f"Crawled [{len(pages)}/{max_pages}]: {title[:50]}...")
            
            # Find links to follow
            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(url, href)
                parsed = urlparse(absolute_url)
                
                # Filter by domain if required
                if same_domain_only and parsed.netloc != domain:
                    continue
                
                # Skip non-doc paths
                skip_patterns = ["/blog", "/news", "/careers", "/about", "/contact", "/pricing"]
                if any(p in parsed.path.lower() for p in skip_patterns):
                    continue
                
                # Prefer doc paths
                if absolute_url not in visited:
                    to_visit.append(absolute_url)
        
        except Exception as e:
            logger.warning(f"Error crawling {url}: {e}")
            continue
    
    logger.info(f"Crawl complete: {len(pages)} pages")
    return pages


# ============================================================================
# Chunking
# ============================================================================

def chunk_page(
    text: str,
    headings_aware: bool = True,
    max_chunk_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Split page text into chunks.
    
    If headings_aware, tries to split at heading boundaries.
    Returns list of {text, heading, chunk_index}
    """
    # Rough token estimation (1 token ≈ 4 chars)
    max_chars = max_chunk_tokens * 4
    overlap_chars = overlap_tokens * 4
    
    chunks = []
    
    if headings_aware:
        # Split by heading patterns
        heading_pattern = r"\n(?=#{1,4}\s|[A-Z][^.!?\n]{5,50}\n[-=]+\n|\d+\.\s[A-Z])"
        sections = re.split(heading_pattern, text)
        
        current_heading = None
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if section starts with a heading
            lines = section.split("\n", 1)
            first_line = lines[0].strip()
            if re.match(r"^#{1,4}\s", first_line) or (len(first_line) < 80 and first_line.isupper()):
                current_heading = first_line.lstrip("#").strip()
            
            # Split long sections
            if len(section) > max_chars:
                sub_chunks = _split_text(section, max_chars, overlap_chars)
                for i, chunk_text in enumerate(sub_chunks):
                    chunks.append({
                        "text": chunk_text,
                        "heading": current_heading,
                        "chunk_index": len(chunks)
                    })
            else:
                chunks.append({
                    "text": section,
                    "heading": current_heading,
                    "chunk_index": len(chunks)
                })
    else:
        # Simple character-based splitting
        sub_chunks = _split_text(text, max_chars, overlap_chars)
        for i, chunk_text in enumerate(sub_chunks):
            chunks.append({
                "text": chunk_text,
                "heading": None,
                "chunk_index": i
            })
    
    return chunks


def _split_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the boundary
            for boundary in [".\n", ". ", "!\n", "! ", "?\n", "? "]:
                last_boundary = text.rfind(boundary, start + max_chars // 2, end)
                if last_boundary > start:
                    end = last_boundary + len(boundary)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap_chars
    
    return chunks


# ============================================================================
# Embedding
# ============================================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Voyage AI.
    
    Returns list of embedding vectors.
    """
    if not texts:
        return []
    
    try:
        import voyageai
        vo = voyageai.Client(api_key=settings.voyage_api_key)
        
        # Batch in groups of 8 to avoid rate limits
        embeddings = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            result = vo.embed(batch, model="voyage-2")
            embeddings.extend(result.embeddings)
        
        return embeddings
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Return zero vectors as fallback
        return [[0.0] * 1024 for _ in texts]


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text."""
    results = embed_texts([text])
    return results[0] if results else [0.0] * 1024


# ============================================================================
# Procedure Extraction
# ============================================================================

def extract_procedures(page_text: str, page_title: str = "") -> List[Dict[str, Any]]:
    """
    Extract procedures (step-by-step instructions) from page text.
    
    Uses heuristics first, with optional LLM fallback.
    
    Returns list of:
    {
        goal: str,
        steps: [
            {idx, instruction, expected_state, action_type, selector_hint}
        ]
    }
    """
    procedures = []
    
    # Pattern 1: Numbered lists (1. 2. 3.)
    numbered_pattern = r"(?:^|\n)(\d+)\.\s+([^\n]+)"
    matches = list(re.finditer(numbered_pattern, page_text))
    
    if len(matches) >= 3:
        # Group consecutive numbered items
        current_procedure = {"steps": []}
        last_num = 0
        
        for match in matches:
            num = int(match.group(1))
            instruction = match.group(2).strip()
            
            if num == 1 or num != last_num + 1:
                # Start of new procedure
                if current_procedure["steps"]:
                    procedures.append(current_procedure)
                current_procedure = {"steps": []}
            
            step = _parse_step(instruction, num)
            current_procedure["steps"].append(step)
            last_num = num
        
        if current_procedure["steps"]:
            procedures.append(current_procedure)
    
    # Pattern 2: "Step X:" patterns
    step_pattern = r"(?:^|\n)(?:Step\s+)?(\d+)[:.]\s*([^\n]+)"
    step_matches = list(re.finditer(step_pattern, page_text, re.IGNORECASE))
    
    if step_matches and not procedures:
        current_procedure = {"steps": []}
        for match in step_matches:
            idx = int(match.group(1))
            instruction = match.group(2).strip()
            step = _parse_step(instruction, idx)
            current_procedure["steps"].append(step)
        
        if current_procedure["steps"]:
            procedures.append(current_procedure)
    
    # Pattern 3: Bullet lists with action verbs
    bullet_pattern = r"(?:^|\n)[\-\*•]\s*((?:Click|Select|Enter|Type|Navigate|Go to|Open|Choose|Press)[^\n]+)"
    bullet_matches = list(re.finditer(bullet_pattern, page_text, re.IGNORECASE))
    
    if len(bullet_matches) >= 3 and not procedures:
        current_procedure = {"steps": []}
        for i, match in enumerate(bullet_matches):
            instruction = match.group(1).strip()
            step = _parse_step(instruction, i + 1)
            current_procedure["steps"].append(step)
        
        if current_procedure["steps"]:
            procedures.append(current_procedure)
    
    # Infer goal for each procedure
    for proc in procedures:
        proc["goal"] = _infer_goal(proc["steps"], page_title)
    
    # Filter out low-quality procedures
    procedures = [p for p in procedures if len(p["steps"]) >= 2 and p["goal"]]
    
    logger.info(f"Extracted {len(procedures)} procedures from page")
    return procedures


def _parse_step(instruction: str, idx: int) -> Dict[str, Any]:
    """Parse a step instruction to extract action type and hints."""
    instruction_lower = instruction.lower()
    
    # Determine action type
    action_type = "unknown"
    selector_hint = None
    
    if any(w in instruction_lower for w in ["click", "select", "press", "tap", "choose"]):
        action_type = "click"
        # Extract selector hint (quoted text or "X button/link")
        quoted = re.search(r'["\']([^"\']+)["\']', instruction)
        button_ref = re.search(r'(?:the\s+)?["\']?(\w+(?:\s+\w+)?)["\']?\s+(?:button|link|tab|menu|option)', instruction, re.I)
        if quoted:
            selector_hint = quoted.group(1)
        elif button_ref:
            selector_hint = button_ref.group(1)
    
    elif any(w in instruction_lower for w in ["type", "enter", "input", "fill", "write"]):
        action_type = "type"
        # Extract what to type
        quoted = re.search(r'["\']([^"\']+)["\']', instruction)
        if quoted:
            selector_hint = quoted.group(1)
    
    elif any(w in instruction_lower for w in ["navigate", "go to", "open", "visit"]):
        action_type = "navigate"
        # Extract URL
        url_match = re.search(r'(https?://[^\s]+)', instruction)
        if url_match:
            selector_hint = url_match.group(1)
    
    elif any(w in instruction_lower for w in ["wait", "pause"]):
        action_type = "wait"
    
    # Infer expected state
    expected_state = None
    if "should see" in instruction_lower or "will see" in instruction_lower:
        state_match = re.search(r'(?:should|will)\s+see\s+(.+)', instruction, re.I)
        if state_match:
            expected_state = state_match.group(1).strip()
    
    return {
        "idx": idx,
        "instruction": instruction,
        "action_type": action_type,
        "selector_hint": selector_hint,
        "expected_state": expected_state
    }


def _infer_goal(steps: List[Dict[str, Any]], page_title: str) -> str:
    """Infer the goal of a procedure from its steps and page title."""
    if page_title:
        # Clean up title
        goal = page_title
        for prefix in ["How to ", "Guide: ", "Tutorial: ", "Docs: "]:
            if goal.startswith(prefix):
                goal = goal[len(prefix):]
        return goal
    
    # Build goal from first action
    if steps:
        first_instruction = steps[0]["instruction"]
        goal = f"Procedure: {first_instruction[:50]}"
        return goal
    
    return "Unknown procedure"


# ============================================================================
# Full Ingestion Pipeline
# ============================================================================

async def ingest_docs_from_url(
    company_id: str,
    root_url: str,
    max_pages: int = 50
) -> Dict[str, Any]:
    """
    Full ingestion pipeline: crawl -> chunk -> embed -> extract -> store.
    
    Returns summary of ingested content.
    """
    logger.info(f"Starting ingestion for company {company_id} from {root_url}")
    
    # Create doc source
    doc_source = graph_service.create_doc_source(
        company_id=company_id,
        source_type="url",
        root_url=root_url
    )
    source_id = doc_source["id"]
    
    try:
        # Crawl pages
        pages = crawl_docs(root_url, max_pages=max_pages)
        
        total_chunks = 0
        total_procedures = 0
        
        for page_data in pages:
            # Create page node
            page = graph_service.create_doc_page(
                source_id=source_id,
                url=page_data["url"],
                title=page_data["title"],
                text=page_data["text"][:10000],  # Limit stored text
                headings=page_data["headings"]
            )
            page_id = page["id"]
            
            # Chunk the page
            chunks = chunk_page(page_data["text"])
            
            # Embed chunks in batches
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_texts(chunk_texts)
            
            # Store chunks
            for chunk, embedding in zip(chunks, embeddings):
                graph_service.create_chunk(
                    page_id=page_id,
                    text=chunk["text"],
                    embedding=embedding,
                    chunk_index=chunk["chunk_index"],
                    heading=chunk.get("heading")
                )
                total_chunks += 1
            
            # Extract and store procedures
            procedures = extract_procedures(page_data["text"], page_data["title"])
            
            for proc_data in procedures:
                # Embed the goal
                goal_embedding = embed_text(proc_data["goal"])
                
                # Create procedure
                procedure = graph_service.create_procedure(
                    page_id=page_id,
                    goal=proc_data["goal"],
                    goal_embedding=goal_embedding,
                    source_text="\n".join([s["instruction"] for s in proc_data["steps"]])
                )
                procedure_id = procedure["id"]
                
                # Create steps
                step_ids = []
                for step_data in proc_data["steps"]:
                    step = graph_service.create_step(
                        procedure_id=procedure_id,
                        step_index=step_data["idx"],
                        instruction=step_data["instruction"],
                        action_type=step_data["action_type"],
                        selector_hint=step_data.get("selector_hint"),
                        expected_state=step_data.get("expected_state")
                    )
                    step_ids.append(step["id"])
                
                # Link steps sequentially
                graph_service.link_steps_sequential(step_ids)
                total_procedures += 1
        
        # Update source status
        graph_service.update_doc_source_status(
            source_id=source_id,
            status="completed",
            page_count=len(pages)
        )
        
        result = {
            "source_id": source_id,
            "pages_crawled": len(pages),
            "chunks_created": total_chunks,
            "procedures_extracted": total_procedures,
            "status": "completed"
        }
        
        logger.info(f"Ingestion complete: {result}")
        return result
    
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        graph_service.update_doc_source_status(source_id, "failed")
        raise


async def ingest_docs_from_file(
    company_id: str,
    filename: str,
    content: bytes,
    content_type: str
) -> Dict[str, Any]:
    """
    Ingest documentation from an uploaded file.
    
    Supports: text/plain, text/markdown, text/html, application/pdf
    """
    logger.info(f"Starting file ingestion for company {company_id}: {filename}")
    
    # Create doc source
    doc_source = graph_service.create_doc_source(
        company_id=company_id,
        source_type="upload",
        filename=filename
    )
    source_id = doc_source["id"]
    
    try:
        # Extract text based on content type
        if content_type == "application/pdf":
            text = _extract_pdf_text(content)
        elif content_type in ["text/html", "application/xhtml+xml"]:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            # Plain text or markdown
            text = content.decode("utf-8", errors="ignore")
        
        # Extract title from first line or filename
        lines = text.split("\n")
        title = lines[0].strip() if lines else filename
        if title.startswith("#"):
            title = title.lstrip("#").strip()
        
        # Extract headings
        headings = []
        for line in lines[:50]:
            if line.startswith("#") or (len(line) < 80 and line.isupper()):
                headings.append(line.lstrip("#").strip())
        
        # Create page node
        page = graph_service.create_doc_page(
            source_id=source_id,
            url=f"file://{filename}",
            title=title,
            text=text[:10000],
            headings=headings
        )
        page_id = page["id"]
        
        # Chunk the content
        chunks = chunk_page(text)
        
        # Embed chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        
        # Store chunks
        for chunk, embedding in zip(chunks, embeddings):
            graph_service.create_chunk(
                page_id=page_id,
                text=chunk["text"],
                embedding=embedding,
                chunk_index=chunk["chunk_index"],
                heading=chunk.get("heading")
            )
        
        # Extract procedures
        procedures = extract_procedures(text, title)
        total_procedures = 0
        
        for proc_data in procedures:
            goal_embedding = embed_text(proc_data["goal"])
            
            procedure = graph_service.create_procedure(
                page_id=page_id,
                goal=proc_data["goal"],
                goal_embedding=goal_embedding,
                source_text="\n".join([s["instruction"] for s in proc_data["steps"]])
            )
            
            step_ids = []
            for step_data in proc_data["steps"]:
                step = graph_service.create_step(
                    procedure_id=procedure["id"],
                    step_index=step_data["idx"],
                    instruction=step_data["instruction"],
                    action_type=step_data["action_type"],
                    selector_hint=step_data.get("selector_hint"),
                    expected_state=step_data.get("expected_state")
                )
                step_ids.append(step["id"])
            
            graph_service.link_steps_sequential(step_ids)
            total_procedures += 1
        
        # Update source status
        graph_service.update_doc_source_status(
            source_id=source_id,
            status="completed",
            page_count=1
        )
        
        return {
            "source_id": source_id,
            "pages_crawled": 1,
            "chunks_created": len(chunks),
            "procedures_extracted": total_procedures,
            "status": "completed"
        }
    
    except Exception as e:
        logger.exception(f"File ingestion failed: {e}")
        graph_service.update_doc_source_status(source_id, "failed")
        raise


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""
