// ============================================================================
// Neo4j Schema for Document Knowledge Base + Commerce
// ============================================================================
// Run these commands in Neo4j Browser or via cypher-shell

// ============================================================================
// CONSTRAINTS - Unique IDs
// ============================================================================

// Document nodes
CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT doc_source_id IF NOT EXISTS FOR (d:DocSource) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT doc_page_id IF NOT EXISTS FOR (p:DocPage) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE;
CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.id IS UNIQUE;
CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT ui_state_id IF NOT EXISTS FOR (u:UIState) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE;

// Commerce nodes
CREATE CONSTRAINT user_profile_id IF NOT EXISTS FOR (u:UserProfile) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT preference_id IF NOT EXISTS FOR (p:Preference) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT nutrition_claim_id IF NOT EXISTS FOR (n:NutritionClaim) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_evidence_id IF NOT EXISTS FOR (e:ProductEvidence) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT comparison_id IF NOT EXISTS FOR (c:Comparison) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT cart_session_id IF NOT EXISTS FOR (c:CartSession) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT purchase_step_id IF NOT EXISTS FOR (p:PurchaseStep) REQUIRE p.id IS UNIQUE;

// ============================================================================
// INDEXES - Fast lookups
// ============================================================================

// Document indexes
CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name);
CREATE INDEX doc_page_url IF NOT EXISTS FOR (p:DocPage) ON (p.url);
CREATE INDEX procedure_goal IF NOT EXISTS FOR (pr:Procedure) ON (pr.goal);
CREATE INDEX chunk_text IF NOT EXISTS FOR (ch:Chunk) ON (ch.text);

// Commerce indexes
CREATE INDEX product_title IF NOT EXISTS FOR (p:Product) ON (p.title);
CREATE INDEX product_handle IF NOT EXISTS FOR (p:Product) ON (p.handle);
CREATE INDEX product_vendor IF NOT EXISTS FOR (p:Product) ON (p.vendor);
CREATE INDEX product_tags IF NOT EXISTS FOR (p:Product) ON (p.tags);
CREATE INDEX user_profile_user_id IF NOT EXISTS FOR (u:UserProfile) ON (u.user_id);

// ============================================================================
// VECTOR INDEXES - Semantic search (requires Neo4j 5.11+)
// ============================================================================

// Vector index for Chunk embeddings
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk)
ON c.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
}};

// Vector index for Product embeddings
CREATE VECTOR INDEX product_embedding IF NOT EXISTS
FOR (p:Product)
ON p.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
}};

// ============================================================================
// FULLTEXT INDEXES (optional - for text search)
// ============================================================================

// Fulltext index for product search
CREATE FULLTEXT INDEX product_fulltext IF NOT EXISTS
FOR (p:Product)
ON EACH [p.title, p.description, p.vendor];

// Fulltext index for chunk search
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.text, c.heading];

// ============================================================================
// RELATIONSHIP SCHEMA (documentation only - Neo4j doesn't enforce)
// ============================================================================

/*
Document relationships:
  (Company)-[:HAS_SOURCE]->(DocSource)
  (DocSource)-[:HAS_PAGE]->(DocPage)
  (DocPage)-[:HAS_CHUNK]->(Chunk)
  (Procedure)-[:DERIVED_FROM]->(DocPage)
  (Procedure)-[:HAS_STEP]->(Step)
  (Step)-[:NEXT]->(Step)
  (Step)-[:REQUIRES_STATE]->(UIState)
  (Step)-[:PRODUCES_STATE]->(UIState)
  (Decision)-[:FOLLOWS]->(Procedure)
  (Decision)-[:JUSTIFIED_BY]->(DocPage|Chunk|Step)

Commerce relationships:
  (UserProfile)-[:HAS_PREFERENCE]->(Preference)
  (Company)-[:SELLS]->(Product)
  (Product)-[:HAS_NUTRITION]->(NutritionClaim)
  (Product)-[:SUPPORTED_BY]->(ProductEvidence)
  (Comparison)-[:BASELINE]->(Product)
  (Comparison)-[:ALTERNATIVE]->(Product)
  (Decision)-[:RECOMMENDS]->(Product)
  (Decision)-[:COMPARES]->(Comparison)
  (Decision)-[:JUSTIFIED_BY]->(ProductEvidence|NutritionClaim|DocPage|Chunk|Step)
  (CartSession)-[:HAS_ITEM]->(Product)
  (Procedure)-[:HAS_STEP]->(PurchaseStep)
*/

// ============================================================================
// VERIFY SCHEMA
// ============================================================================

// Show all constraints
SHOW CONSTRAINTS;

// Show all indexes
SHOW INDEXES;
