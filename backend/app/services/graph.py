"""
Neo4j Graph Service for Document Knowledge Base + Commerce

Schema:
- Nodes: Company, DocSource, DocPage, Chunk, Procedure, Step, UIState, Decision
         UserProfile, Preference, Product, NutritionClaim, ProductEvidence,
         Comparison, CartSession, PurchaseStep
- Relationships: HAS_SOURCE, HAS_PAGE, HAS_CHUNK, DERIVED_FROM, HAS_STEP, NEXT,
                 REQUIRES_STATE, PRODUCES_STATE, FOLLOWS, JUSTIFIED_BY,
                 HAS_PREFERENCE, SELLS, HAS_NUTRITION, SUPPORTED_BY, BASELINE,
                 ALTERNATIVE, RECOMMENDS, COMPARES, HAS_ITEM
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable

from app.config import settings

logger = logging.getLogger(__name__)


class GraphService:
    """Neo4j graph database service for document knowledge base."""
    
    _driver: Optional[Driver] = None
    
    @classmethod
    def get_driver(cls) -> Driver:
        """Get or create Neo4j driver."""
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")
        return cls._driver
    
    @classmethod
    def close(cls) -> None:
        """Close the Neo4j driver."""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Closed Neo4j connection")
    
    @classmethod
    def verify_connectivity(cls) -> bool:
        """Verify Neo4j connection."""
        try:
            driver = cls.get_driver()
            driver.verify_connectivity()
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
    
    @classmethod
    def setup_schema(cls) -> None:
        """Create constraints and indexes for the knowledge base schema."""
        driver = cls.get_driver()
        
        constraints = [
            # Unique constraints - Document nodes
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT doc_source_id IF NOT EXISTS FOR (d:DocSource) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT doc_page_id IF NOT EXISTS FOR (p:DocPage) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE",
            "CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.id IS UNIQUE",
            "CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT ui_state_id IF NOT EXISTS FOR (u:UIState) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            # Unique constraints - Commerce nodes
            "CREATE CONSTRAINT user_profile_id IF NOT EXISTS FOR (u:UserProfile) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT preference_id IF NOT EXISTS FOR (p:Preference) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT nutrition_claim_id IF NOT EXISTS FOR (n:NutritionClaim) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT product_evidence_id IF NOT EXISTS FOR (e:ProductEvidence) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT comparison_id IF NOT EXISTS FOR (c:Comparison) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT cart_session_id IF NOT EXISTS FOR (c:CartSession) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT purchase_step_id IF NOT EXISTS FOR (p:PurchaseStep) REQUIRE p.id IS UNIQUE",
        ]
        
        indexes = [
            # Text indexes for search - Document nodes
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX doc_page_url IF NOT EXISTS FOR (p:DocPage) ON (p.url)",
            "CREATE INDEX procedure_goal IF NOT EXISTS FOR (pr:Procedure) ON (pr.goal)",
            "CREATE INDEX chunk_text IF NOT EXISTS FOR (ch:Chunk) ON (ch.text)",
            # Indexes - Commerce nodes
            "CREATE INDEX product_title IF NOT EXISTS FOR (p:Product) ON (p.title)",
            "CREATE INDEX product_handle IF NOT EXISTS FOR (p:Product) ON (p.handle)",
            "CREATE INDEX product_vendor IF NOT EXISTS FOR (p:Product) ON (p.vendor)",
            "CREATE INDEX product_tags IF NOT EXISTS FOR (p:Product) ON (p.tags)",
            "CREATE INDEX user_profile_user_id IF NOT EXISTS FOR (u:UserProfile) ON (u.user_id)",
            # Note: Vector index for embeddings created separately
        ]
        
        with driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
        
        logger.info("Neo4j schema setup complete")
    
    @classmethod
    def setup_vector_index(cls) -> None:
        """Create vector indexes for embeddings (requires Neo4j 5.11+)."""
        driver = cls.get_driver()
        
        vector_indexes = [
            # Vector index for Chunk embeddings
            """
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            # Vector index for Product embeddings
            """
            CREATE VECTOR INDEX product_embedding IF NOT EXISTS
            FOR (p:Product)
            ON p.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}
            """,
        ]
        
        try:
            with driver.session() as session:
                for query in vector_indexes:
                    session.run(query)
            logger.info("Vector indexes created (Chunk + Product embeddings)")
        except Exception as e:
            logger.warning(f"Vector index creation failed (may need Neo4j 5.11+): {e}")

    # =========================================================================
    # Company CRUD
    # =========================================================================
    
    @classmethod
    def create_company(cls, name: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Create a new company node."""
        driver = cls.get_driver()
        company_id = str(uuid.uuid4())
        
        query = """
        CREATE (c:Company {
            id: $id,
            name: $name,
            domain: $domain,
            created_at: datetime()
        })
        RETURN c
        """
        
        with driver.session() as session:
            result = session.run(query, id=company_id, name=name, domain=domain)
            record = result.single()
            return dict(record["c"]) if record else None
    
    @classmethod
    def get_company(cls, company_id: str) -> Optional[Dict[str, Any]]:
        """Get a company by ID."""
        driver = cls.get_driver()
        
        query = "MATCH (c:Company {id: $id}) RETURN c"
        
        with driver.session() as session:
            result = session.run(query, id=company_id)
            record = result.single()
            return dict(record["c"]) if record else None
    
    @classmethod
    def list_companies(cls) -> List[Dict[str, Any]]:
        """List all companies."""
        driver = cls.get_driver()
        
        query = "MATCH (c:Company) RETURN c ORDER BY c.created_at DESC"
        
        with driver.session() as session:
            result = session.run(query)
            return [dict(record["c"]) for record in result]

    # =========================================================================
    # DocSource CRUD
    # =========================================================================
    
    @classmethod
    def create_doc_source(
        cls,
        company_id: str,
        source_type: str,  # "url" | "upload"
        root_url: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a documentation source linked to a company."""
        driver = cls.get_driver()
        source_id = str(uuid.uuid4())
        
        query = """
        MATCH (c:Company {id: $company_id})
        CREATE (d:DocSource {
            id: $id,
            source_type: $source_type,
            root_url: $root_url,
            filename: $filename,
            created_at: datetime(),
            status: 'pending'
        })
        CREATE (c)-[:HAS_SOURCE]->(d)
        RETURN d
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                company_id=company_id,
                id=source_id,
                source_type=source_type,
                root_url=root_url,
                filename=filename
            )
            record = result.single()
            return dict(record["d"]) if record else None
    
    @classmethod
    def update_doc_source_status(cls, source_id: str, status: str, page_count: int = 0) -> None:
        """Update the status of a doc source."""
        driver = cls.get_driver()
        
        query = """
        MATCH (d:DocSource {id: $id})
        SET d.status = $status, d.page_count = $page_count, d.updated_at = datetime()
        """
        
        with driver.session() as session:
            session.run(query, id=source_id, status=status, page_count=page_count)

    # =========================================================================
    # DocPage CRUD
    # =========================================================================
    
    @classmethod
    def create_doc_page(
        cls,
        source_id: str,
        url: str,
        title: str,
        text: str,
        headings: List[str]
    ) -> Dict[str, Any]:
        """Create a documentation page linked to a source."""
        driver = cls.get_driver()
        page_id = str(uuid.uuid4())
        
        query = """
        MATCH (s:DocSource {id: $source_id})
        CREATE (p:DocPage {
            id: $id,
            url: $url,
            title: $title,
            text: $text,
            headings: $headings,
            created_at: datetime()
        })
        CREATE (s)-[:HAS_PAGE]->(p)
        RETURN p
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                source_id=source_id,
                id=page_id,
                url=url,
                title=title,
                text=text,
                headings=headings
            )
            record = result.single()
            return dict(record["p"]) if record else None

    # =========================================================================
    # Chunk CRUD
    # =========================================================================
    
    @classmethod
    def create_chunk(
        cls,
        page_id: str,
        text: str,
        embedding: List[float],
        chunk_index: int,
        heading: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a text chunk with embedding linked to a page."""
        driver = cls.get_driver()
        chunk_id = str(uuid.uuid4())
        
        query = """
        MATCH (p:DocPage {id: $page_id})
        CREATE (ch:Chunk {
            id: $id,
            text: $text,
            embedding: $embedding,
            chunk_index: $chunk_index,
            heading: $heading,
            created_at: datetime()
        })
        CREATE (p)-[:HAS_CHUNK]->(ch)
        RETURN ch.id AS id
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                page_id=page_id,
                id=chunk_id,
                text=text,
                embedding=embedding,
                chunk_index=chunk_index,
                heading=heading
            )
            record = result.single()
            return {"id": record["id"]} if record else None

    # =========================================================================
    # Procedure & Step CRUD
    # =========================================================================
    
    @classmethod
    def create_procedure(
        cls,
        page_id: str,
        goal: str,
        goal_embedding: List[float],
        source_text: str
    ) -> Dict[str, Any]:
        """Create a procedure derived from a documentation page."""
        driver = cls.get_driver()
        procedure_id = str(uuid.uuid4())
        
        query = """
        MATCH (p:DocPage {id: $page_id})
        CREATE (pr:Procedure {
            id: $id,
            goal: $goal,
            goal_embedding: $goal_embedding,
            source_text: $source_text,
            created_at: datetime()
        })
        CREATE (pr)-[:DERIVED_FROM]->(p)
        RETURN pr
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                page_id=page_id,
                id=procedure_id,
                goal=goal,
                goal_embedding=goal_embedding,
                source_text=source_text
            )
            record = result.single()
            return dict(record["pr"]) if record else None
    
    @classmethod
    def create_step(
        cls,
        procedure_id: str,
        step_index: int,
        instruction: str,
        action_type: str,  # "click" | "type" | "navigate" | "wait"
        selector_hint: Optional[str] = None,
        expected_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a step within a procedure."""
        driver = cls.get_driver()
        step_id = str(uuid.uuid4())
        
        query = """
        MATCH (pr:Procedure {id: $procedure_id})
        CREATE (s:Step {
            id: $id,
            step_index: $step_index,
            instruction: $instruction,
            action_type: $action_type,
            selector_hint: $selector_hint,
            expected_state: $expected_state,
            created_at: datetime()
        })
        CREATE (pr)-[:HAS_STEP]->(s)
        RETURN s
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                procedure_id=procedure_id,
                id=step_id,
                step_index=step_index,
                instruction=instruction,
                action_type=action_type,
                selector_hint=selector_hint,
                expected_state=expected_state
            )
            record = result.single()
            return dict(record["s"]) if record else None
    
    @classmethod
    def link_steps_sequential(cls, step_ids: List[str]) -> None:
        """Create NEXT relationships between sequential steps."""
        driver = cls.get_driver()
        
        if len(step_ids) < 2:
            return
        
        query = """
        MATCH (s1:Step {id: $from_id})
        MATCH (s2:Step {id: $to_id})
        CREATE (s1)-[:NEXT]->(s2)
        """
        
        with driver.session() as session:
            for i in range(len(step_ids) - 1):
                session.run(query, from_id=step_ids[i], to_id=step_ids[i + 1])
    
    @classmethod
    def create_ui_state(
        cls,
        description: str,
        url_pattern: Optional[str] = None,
        element_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a UI state node."""
        driver = cls.get_driver()
        state_id = str(uuid.uuid4())
        
        query = """
        CREATE (u:UIState {
            id: $id,
            description: $description,
            url_pattern: $url_pattern,
            element_hints: $element_hints,
            created_at: datetime()
        })
        RETURN u
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                id=state_id,
                description=description,
                url_pattern=url_pattern,
                element_hints=element_hints or []
            )
            record = result.single()
            return dict(record["u"]) if record else None
    
    @classmethod
    def link_step_states(
        cls,
        step_id: str,
        requires_state_id: Optional[str] = None,
        produces_state_id: Optional[str] = None
    ) -> None:
        """Link a step to required/produced UI states."""
        driver = cls.get_driver()
        
        with driver.session() as session:
            if requires_state_id:
                session.run("""
                    MATCH (s:Step {id: $step_id})
                    MATCH (u:UIState {id: $state_id})
                    CREATE (s)-[:REQUIRES_STATE]->(u)
                """, step_id=step_id, state_id=requires_state_id)
            
            if produces_state_id:
                session.run("""
                    MATCH (s:Step {id: $step_id})
                    MATCH (u:UIState {id: $state_id})
                    CREATE (s)-[:PRODUCES_STATE]->(u)
                """, step_id=step_id, state_id=produces_state_id)

    # =========================================================================
    # Decision Trace
    # =========================================================================
    
    @classmethod
    def create_decision(
        cls,
        session_id: str,
        action_type: str,
        action_data: Dict[str, Any],
        procedure_id: Optional[str] = None,
        step_id: Optional[str] = None,
        justification_ids: Optional[List[str]] = None  # DocPage, Chunk, or Step IDs
    ) -> Dict[str, Any]:
        """Create a decision trace node with justification links."""
        driver = cls.get_driver()
        decision_id = str(uuid.uuid4())
        
        # Create the decision node
        query = """
        CREATE (d:Decision {
            id: $id,
            session_id: $session_id,
            action_type: $action_type,
            action_data: $action_data,
            created_at: datetime()
        })
        RETURN d
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                id=decision_id,
                session_id=session_id,
                action_type=action_type,
                action_data=str(action_data)
            )
            record = result.single()
            
            # Link to procedure if provided
            if procedure_id:
                session.run("""
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (pr:Procedure {id: $procedure_id})
                    CREATE (d)-[:FOLLOWS]->(pr)
                """, decision_id=decision_id, procedure_id=procedure_id)
            
            # Link to step if provided
            if step_id:
                session.run("""
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (s:Step {id: $step_id})
                    CREATE (d)-[:JUSTIFIED_BY]->(s)
                """, decision_id=decision_id, step_id=step_id)
            
            # Link to justification sources
            if justification_ids:
                for jid in justification_ids:
                    session.run("""
                        MATCH (d:Decision {id: $decision_id})
                        MATCH (j) WHERE j.id = $jid AND (j:DocPage OR j:Chunk OR j:Step)
                        CREATE (d)-[:JUSTIFIED_BY]->(j)
                    """, decision_id=decision_id, jid=jid)
            
            return dict(record["d"]) if record else None

    # =========================================================================
    # Query Methods
    # =========================================================================
    
    @classmethod
    def get_procedures_for_company(cls, company_id: str) -> List[Dict[str, Any]]:
        """Get all procedures for a company with their steps."""
        driver = cls.get_driver()
        
        query = """
        MATCH (c:Company {id: $company_id})-[:HAS_SOURCE]->(s:DocSource)-[:HAS_PAGE]->(p:DocPage)<-[:DERIVED_FROM]-(pr:Procedure)
        OPTIONAL MATCH (pr)-[:HAS_STEP]->(step:Step)
        WITH pr, p, collect(step) AS steps
        RETURN pr, p.url AS source_url, p.title AS source_title,
               [s IN steps | {id: s.id, index: s.step_index, instruction: s.instruction, action_type: s.action_type}] AS steps
        ORDER BY pr.created_at DESC
        """
        
        with driver.session() as session:
            result = session.run(query, company_id=company_id)
            procedures = []
            for record in result:
                proc = dict(record["pr"])
                proc["source_url"] = record["source_url"]
                proc["source_title"] = record["source_title"]
                proc["steps"] = sorted(record["steps"], key=lambda x: x.get("index", 0))
                procedures.append(proc)
            return procedures
    
    @classmethod
    def find_similar_procedures(
        cls,
        company_id: str,
        goal_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find procedures similar to a goal using vector similarity."""
        driver = cls.get_driver()
        
        # Use vector index if available, otherwise fallback to brute force
        query = """
        MATCH (c:Company {id: $company_id})-[:HAS_SOURCE]->(:DocSource)-[:HAS_PAGE]->(:DocPage)<-[:DERIVED_FROM]-(pr:Procedure)
        WHERE pr.goal_embedding IS NOT NULL
        WITH pr, gds.similarity.cosine(pr.goal_embedding, $embedding) AS score
        WHERE score > 0.5
        RETURN pr, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        # Fallback query without GDS (basic similarity)
        fallback_query = """
        MATCH (c:Company {id: $company_id})-[:HAS_SOURCE]->(:DocSource)-[:HAS_PAGE]->(:DocPage)<-[:DERIVED_FROM]-(pr:Procedure)
        RETURN pr
        LIMIT $limit
        """
        
        with driver.session() as session:
            try:
                result = session.run(query, company_id=company_id, embedding=goal_embedding, limit=limit)
                return [{"procedure": dict(r["pr"]), "score": r["score"]} for r in result]
            except Exception:
                # GDS not available, use fallback
                result = session.run(fallback_query, company_id=company_id, limit=limit)
                return [{"procedure": dict(r["pr"]), "score": 1.0} for r in result]
    
    @classmethod
    def find_similar_chunks(
        cls,
        company_id: str,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to a query using vector similarity."""
        driver = cls.get_driver()
        
        query = """
        MATCH (c:Company {id: $company_id})-[:HAS_SOURCE]->(:DocSource)-[:HAS_PAGE]->(p:DocPage)-[:HAS_CHUNK]->(ch:Chunk)
        WHERE ch.embedding IS NOT NULL
        RETURN ch, p.url AS page_url, p.title AS page_title
        LIMIT $limit
        """
        
        with driver.session() as session:
            result = session.run(query, company_id=company_id, limit=limit)
            chunks = []
            for record in result:
                chunk = dict(record["ch"])
                chunk["page_url"] = record["page_url"]
                chunk["page_title"] = record["page_title"]
                chunks.append(chunk)
            return chunks
    
    @classmethod
    def get_procedure_with_steps(cls, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get a procedure with all its steps in order."""
        driver = cls.get_driver()
        
        query = """
        MATCH (pr:Procedure {id: $procedure_id})
        OPTIONAL MATCH (pr)-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (pr)-[:DERIVED_FROM]->(p:DocPage)
        WITH pr, p, collect(s) AS steps
        RETURN pr, p, steps
        """
        
        with driver.session() as session:
            result = session.run(query, procedure_id=procedure_id)
            record = result.single()
            if not record:
                return None
            
            proc = dict(record["pr"])
            proc["source_page"] = dict(record["p"]) if record["p"] else None
            proc["steps"] = sorted(
                [dict(s) for s in record["steps"]],
                key=lambda x: x.get("step_index", 0)
            )
            return proc


    # =========================================================================
    # Commerce: UserProfile & Preferences
    # =========================================================================
    
    @classmethod
    def create_user_profile(
        cls,
        user_id: str,
        age_group: Optional[str] = None  # "minor", "adult", "unknown"
    ) -> Dict[str, Any]:
        """Create a user profile."""
        driver = cls.get_driver()
        profile_id = str(uuid.uuid4())
        
        query = """
        CREATE (u:UserProfile {
            id: $id,
            user_id: $user_id,
            age_group: $age_group,
            created_at: datetime()
        })
        RETURN u
        """
        
        with driver.session() as session:
            result = session.run(query, id=profile_id, user_id=user_id, age_group=age_group or "unknown")
            record = result.single()
            return dict(record["u"]) if record else None
    
    @classmethod
    def get_user_profile(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by user_id."""
        driver = cls.get_driver()
        
        query = """
        MATCH (u:UserProfile {user_id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
        RETURN u, collect(p) AS preferences
        """
        
        with driver.session() as session:
            result = session.run(query, user_id=user_id)
            record = result.single()
            if not record:
                return None
            profile = dict(record["u"])
            profile["preferences"] = [dict(p) for p in record["preferences"]]
            return profile
    
    @classmethod
    def add_user_preference(
        cls,
        user_id: str,
        pref_type: str,  # "allergy", "diet", "budget", "taste", "calorie_limit", "sugar_limit"
        value: str
    ) -> Dict[str, Any]:
        """Add a preference to a user profile."""
        driver = cls.get_driver()
        pref_id = str(uuid.uuid4())
        
        query = """
        MATCH (u:UserProfile {user_id: $user_id})
        CREATE (p:Preference {
            id: $pref_id,
            type: $pref_type,
            value: $value,
            created_at: datetime()
        })
        CREATE (u)-[:HAS_PREFERENCE]->(p)
        RETURN p
        """
        
        with driver.session() as session:
            result = session.run(query, user_id=user_id, pref_id=pref_id, pref_type=pref_type, value=value)
            record = result.single()
            return dict(record["p"]) if record else None
    
    @classmethod
    def clear_user_preferences(cls, user_id: str, pref_type: Optional[str] = None) -> int:
        """Clear user preferences, optionally by type."""
        driver = cls.get_driver()
        
        if pref_type:
            query = """
            MATCH (u:UserProfile {user_id: $user_id})-[:HAS_PREFERENCE]->(p:Preference {type: $pref_type})
            DETACH DELETE p
            RETURN count(p) AS deleted
            """
            params = {"user_id": user_id, "pref_type": pref_type}
        else:
            query = """
            MATCH (u:UserProfile {user_id: $user_id})-[:HAS_PREFERENCE]->(p:Preference)
            DETACH DELETE p
            RETURN count(p) AS deleted
            """
            params = {"user_id": user_id}
        
        with driver.session() as session:
            result = session.run(query, **params)
            record = result.single()
            return record["deleted"] if record else 0

    # =========================================================================
    # Commerce: Products
    # =========================================================================
    
    @classmethod
    def create_product(
        cls,
        company_id: str,
        title: str,
        handle: str,
        vendor: Optional[str] = None,
        price: Optional[float] = None,
        currency: str = "CAD",
        product_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        ingredients: Optional[str] = None,
        url: Optional[str] = None,
        image_url: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        shopify_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a product linked to a company."""
        driver = cls.get_driver()
        product_id = str(uuid.uuid4())
        
        query = """
        MATCH (c:Company {id: $company_id})
        CREATE (p:Product {
            id: $id,
            shopify_id: $shopify_id,
            title: $title,
            handle: $handle,
            vendor: $vendor,
            price: $price,
            currency: $currency,
            product_type: $product_type,
            tags: $tags,
            description: $description,
            ingredients: $ingredients,
            url: $url,
            image_url: $image_url,
            embedding: $embedding,
            created_at: datetime()
        })
        CREATE (c)-[:SELLS]->(p)
        RETURN p
        """
        
        with driver.session() as session:
            result = session.run(
                query,
                company_id=company_id,
                id=product_id,
                shopify_id=shopify_id,
                title=title,
                handle=handle,
                vendor=vendor,
                price=price,
                currency=currency,
                product_type=product_type,
                tags=tags or [],
                description=description,
                ingredients=ingredients,
                url=url,
                image_url=image_url,
                embedding=embedding
            )
            record = result.single()
            return dict(record["p"]) if record else None
    
    @classmethod
    def get_product(cls, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a product by ID with nutrition claims and evidence."""
        driver = cls.get_driver()
        
        query = """
        MATCH (p:Product {id: $product_id})
        OPTIONAL MATCH (p)-[:HAS_NUTRITION]->(n:NutritionClaim)
        OPTIONAL MATCH (p)-[:SUPPORTED_BY]->(e:ProductEvidence)
        RETURN p, collect(DISTINCT n) AS nutrition, collect(DISTINCT e) AS evidence
        """
        
        with driver.session() as session:
            result = session.run(query, product_id=product_id)
            record = result.single()
            if not record:
                return None
            product = dict(record["p"])
            product["nutrition_claims"] = [dict(n) for n in record["nutrition"]]
            product["evidence"] = [dict(e) for e in record["evidence"]]
            return product
    
    @classmethod
    def get_product_by_handle(cls, company_id: str, handle: str) -> Optional[Dict[str, Any]]:
        """Get a product by handle within a company."""
        driver = cls.get_driver()
        
        query = """
        MATCH (c:Company {id: $company_id})-[:SELLS]->(p:Product {handle: $handle})
        RETURN p
        """
        
        with driver.session() as session:
            result = session.run(query, company_id=company_id, handle=handle)
            record = result.single()
            return dict(record["p"]) if record else None
    
    @classmethod
    def find_products_by_category(
        cls,
        company_id: str,
        product_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find products by category/tags."""
        driver = cls.get_driver()
        
        if tags:
            query = """
            MATCH (c:Company {id: $company_id})-[:SELLS]->(p:Product)
            WHERE any(tag IN $tags WHERE tag IN p.tags) OR p.product_type = $product_type
            RETURN p
            LIMIT $limit
            """
        else:
            query = """
            MATCH (c:Company {id: $company_id})-[:SELLS]->(p:Product)
            WHERE p.product_type = $product_type OR $product_type IS NULL
            RETURN p
            LIMIT $limit
            """
        
        with driver.session() as session:
            result = session.run(query, company_id=company_id, product_type=product_type, tags=tags or [], limit=limit)
            return [dict(record["p"]) for record in result]
    
    @classmethod
    def find_similar_products(
        cls,
        company_id: str,
        query_embedding: List[float],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find products similar to a query using vector similarity."""
        driver = cls.get_driver()
        
        # Fallback: return products with embeddings
        query = """
        MATCH (c:Company {id: $company_id})-[:SELLS]->(p:Product)
        WHERE p.embedding IS NOT NULL
        RETURN p
        LIMIT $limit
        """
        
        with driver.session() as session:
            result = session.run(query, company_id=company_id, limit=limit)
            return [dict(record["p"]) for record in result]

    # =========================================================================
    # Commerce: NutritionClaim
    # =========================================================================
    
    @classmethod
    def add_nutrition_claim(
        cls,
        product_id: str,
        metric: str,  # "sugar_g", "fiber_g", "protein_g", "calories", etc.
        value: float,
        unit: str = "g",
        basis: str = "per_serving"
    ) -> Dict[str, Any]:
        """Add a nutrition claim to a product."""
        driver = cls.get_driver()
        claim_id = str(uuid.uuid4())
        
        query = """
        MATCH (p:Product {id: $product_id})
        CREATE (n:NutritionClaim {
            id: $claim_id,
            metric: $metric,
            value: $value,
            unit: $unit,
            basis: $basis,
            created_at: datetime()
        })
        CREATE (p)-[:HAS_NUTRITION]->(n)
        RETURN n
        """
        
        with driver.session() as session:
            result = session.run(
                query, product_id=product_id, claim_id=claim_id,
                metric=metric, value=value, unit=unit, basis=basis
            )
            record = result.single()
            return dict(record["n"]) if record else None
    
    @classmethod
    def get_product_nutrition(cls, product_id: str) -> List[Dict[str, Any]]:
        """Get all nutrition claims for a product."""
        driver = cls.get_driver()
        
        query = """
        MATCH (p:Product {id: $product_id})-[:HAS_NUTRITION]->(n:NutritionClaim)
        RETURN n
        """
        
        with driver.session() as session:
            result = session.run(query, product_id=product_id)
            return [dict(record["n"]) for record in result]

    # =========================================================================
    # Commerce: ProductEvidence
    # =========================================================================
    
    @classmethod
    def add_product_evidence(
        cls,
        product_id: str,
        source_type: str,  # "web_page", "product_page", "review", "label"
        source_ref: str,  # URL or reference
        snippet: str,
        fetched_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Add evidence supporting a product claim."""
        driver = cls.get_driver()
        evidence_id = str(uuid.uuid4())
        
        query = """
        MATCH (p:Product {id: $product_id})
        CREATE (e:ProductEvidence {
            id: $evidence_id,
            source_type: $source_type,
            source_ref: $source_ref,
            snippet: $snippet,
            fetched_at: $fetched_at,
            created_at: datetime()
        })
        CREATE (p)-[:SUPPORTED_BY]->(e)
        RETURN e
        """
        
        with driver.session() as session:
            result = session.run(
                query, product_id=product_id, evidence_id=evidence_id,
                source_type=source_type, source_ref=source_ref, snippet=snippet,
                fetched_at=fetched_at or datetime.utcnow()
            )
            record = result.single()
            return dict(record["e"]) if record else None

    # =========================================================================
    # Commerce: Comparison
    # =========================================================================
    
    @classmethod
    def create_comparison(
        cls,
        baseline_product_id: str,
        alternative_product_id: str,
        reason_summary: str
    ) -> Dict[str, Any]:
        """Create a comparison between baseline and alternative products."""
        driver = cls.get_driver()
        comparison_id = str(uuid.uuid4())
        
        query = """
        MATCH (baseline:Product {id: $baseline_id})
        MATCH (alternative:Product {id: $alternative_id})
        CREATE (c:Comparison {
            id: $comparison_id,
            reason_summary: $reason_summary,
            created_at: datetime()
        })
        CREATE (c)-[:BASELINE]->(baseline)
        CREATE (c)-[:ALTERNATIVE]->(alternative)
        RETURN c
        """
        
        with driver.session() as session:
            result = session.run(
                query, baseline_id=baseline_product_id, alternative_id=alternative_product_id,
                comparison_id=comparison_id, reason_summary=reason_summary
            )
            record = result.single()
            return dict(record["c"]) if record else None

    # =========================================================================
    # Commerce: CartSession & PurchaseStep
    # =========================================================================
    
    @classmethod
    def create_cart_session(
        cls,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a cart session."""
        driver = cls.get_driver()
        cart_id = session_id or str(uuid.uuid4())
        
        query = """
        CREATE (c:CartSession {
            id: $cart_id,
            user_id: $user_id,
            status: 'active',
            created_at: datetime()
        })
        RETURN c
        """
        
        with driver.session() as session:
            result = session.run(query, cart_id=cart_id, user_id=user_id)
            record = result.single()
            return dict(record["c"]) if record else None
    
    @classmethod
    def add_item_to_cart(cls, cart_id: str, product_id: str, quantity: int = 1) -> bool:
        """Add a product to a cart session."""
        driver = cls.get_driver()
        
        query = """
        MATCH (c:CartSession {id: $cart_id})
        MATCH (p:Product {id: $product_id})
        MERGE (c)-[r:HAS_ITEM]->(p)
        SET r.quantity = COALESCE(r.quantity, 0) + $quantity
        RETURN r
        """
        
        with driver.session() as session:
            result = session.run(query, cart_id=cart_id, product_id=product_id, quantity=quantity)
            return result.single() is not None
    
    @classmethod
    def get_cart_items(cls, cart_id: str) -> List[Dict[str, Any]]:
        """Get all items in a cart."""
        driver = cls.get_driver()
        
        query = """
        MATCH (c:CartSession {id: $cart_id})-[r:HAS_ITEM]->(p:Product)
        RETURN p, r.quantity AS quantity
        """
        
        with driver.session() as session:
            result = session.run(query, cart_id=cart_id)
            return [{"product": dict(record["p"]), "quantity": record["quantity"]} for record in result]
    
    @classmethod
    def create_purchase_procedure(cls, company_id: str) -> Dict[str, Any]:
        """Create a standard purchase procedure for a company."""
        driver = cls.get_driver()
        procedure_id = str(uuid.uuid4())
        
        # Standard purchase steps
        steps = [
            {"name": "search_product", "instruction": "Search for the product", "action_type": "type"},
            {"name": "open_product", "instruction": "Open the product page", "action_type": "click"},
            {"name": "add_to_cart", "instruction": "Click Add to Cart button", "action_type": "click"},
            {"name": "view_cart", "instruction": "Go to cart", "action_type": "click"},
            {"name": "checkout_shipping", "instruction": "Enter shipping information", "action_type": "type"},
            {"name": "checkout_payment", "instruction": "Enter payment information", "action_type": "type"},
            {"name": "review_order", "instruction": "Review order before confirmation - STOP AND CONFIRM", "action_type": "wait"},
        ]
        
        with driver.session() as session:
            # Create procedure
            proc_query = """
            MATCH (c:Company {id: $company_id})
            CREATE (pr:Procedure {
                id: $procedure_id,
                goal: 'Complete product purchase',
                procedure_type: 'purchase',
                created_at: datetime()
            })
            CREATE (c)-[:HAS_PROCEDURE]->(pr)
            RETURN pr
            """
            session.run(proc_query, company_id=company_id, procedure_id=procedure_id)
            
            # Create steps
            step_ids = []
            for idx, step_data in enumerate(steps):
                step_id = str(uuid.uuid4())
                step_query = """
                MATCH (pr:Procedure {id: $procedure_id})
                CREATE (s:PurchaseStep {
                    id: $step_id,
                    name: $name,
                    step_index: $idx,
                    instruction: $instruction,
                    action_type: $action_type,
                    requires_confirmation: $requires_confirmation,
                    created_at: datetime()
                })
                CREATE (pr)-[:HAS_STEP]->(s)
                RETURN s
                """
                requires_confirmation = step_data["name"] == "review_order"
                session.run(
                    step_query, procedure_id=procedure_id, step_id=step_id,
                    name=step_data["name"], idx=idx + 1, instruction=step_data["instruction"],
                    action_type=step_data["action_type"], requires_confirmation=requires_confirmation
                )
                step_ids.append(step_id)
            
            # Link steps
            cls.link_steps_sequential(step_ids)
            
            return {"id": procedure_id, "step_count": len(steps)}

    # =========================================================================
    # Commerce: Decision Extensions
    # =========================================================================
    
    @classmethod
    def create_recommendation_decision(
        cls,
        session_id: str,
        user_id: str,
        baseline_product_id: Optional[str],
        recommended_product_ids: List[str],
        comparison_ids: List[str],
        evidence_ids: List[str],
        reasoning: str
    ) -> Dict[str, Any]:
        """Create a decision trace for a product recommendation."""
        driver = cls.get_driver()
        decision_id = str(uuid.uuid4())
        
        with driver.session() as session:
            # Create decision
            decision_query = """
            CREATE (d:Decision {
                id: $decision_id,
                session_id: $session_id,
                action_type: 'recommend',
                reasoning: $reasoning,
                created_at: datetime()
            })
            RETURN d
            """
            session.run(decision_query, decision_id=decision_id, session_id=session_id, reasoning=reasoning)
            
            # Link to recommended products
            for product_id in recommended_product_ids:
                session.run("""
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (p:Product {id: $product_id})
                    CREATE (d)-[:RECOMMENDS]->(p)
                """, decision_id=decision_id, product_id=product_id)
            
            # Link to comparisons
            for comparison_id in comparison_ids:
                session.run("""
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (c:Comparison {id: $comparison_id})
                    CREATE (d)-[:COMPARES]->(c)
                """, decision_id=decision_id, comparison_id=comparison_id)
            
            # Link to evidence
            for evidence_id in evidence_ids:
                session.run("""
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (e:ProductEvidence {id: $evidence_id})
                    CREATE (d)-[:JUSTIFIED_BY]->(e)
                """, decision_id=decision_id, evidence_id=evidence_id)
            
            return {"id": decision_id}
    
    @classmethod
    def get_decision_trace(cls, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a decision with all its justifications."""
        driver = cls.get_driver()
        
        query = """
        MATCH (d:Decision {id: $decision_id})
        OPTIONAL MATCH (d)-[:RECOMMENDS]->(p:Product)
        OPTIONAL MATCH (d)-[:COMPARES]->(c:Comparison)
        OPTIONAL MATCH (d)-[:JUSTIFIED_BY]->(j)
        OPTIONAL MATCH (d)-[:FOLLOWS]->(pr:Procedure)
        RETURN d, 
               collect(DISTINCT p) AS products,
               collect(DISTINCT c) AS comparisons,
               collect(DISTINCT j) AS justifications,
               pr
        """
        
        with driver.session() as session:
            result = session.run(query, decision_id=decision_id)
            record = result.single()
            if not record:
                return None
            
            decision = dict(record["d"])
            decision["recommended_products"] = [dict(p) for p in record["products"]]
            decision["comparisons"] = [dict(c) for c in record["comparisons"]]
            decision["justifications"] = [dict(j) for j in record["justifications"]]
            decision["procedure"] = dict(record["pr"]) if record["pr"] else None
            return decision


# Singleton instance
graph_service = GraphService()
