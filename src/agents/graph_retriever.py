"""
src/agents/graph_retriever.py

Knowledge graph retriever stub for Neo4j integration.

This module provides the interface for querying the skincare knowledge graph.
Full implementation requires a running Neo4j instance with data seeded via
scripts/seed_graph.py.

Architecture position: Layer 2.5 — between the vector knowledge base (Layer 2)
and the AI agents (Layer 3). Augments RAG retrieval results with structured
graph facts about ingredient interactions, contraindications, and treatments.

Usage (once Neo4j is running):
    retriever = GraphRetriever(neo4j_uri="bolt://localhost:7687", ...)
    facts = retriever.get_ingredient_relations("retinol")
    augmented = retriever.augment_retrieval_results(rag_results, profile)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphFact:
    """
    A structured fact retrieved from the knowledge graph.

    Example:
        GraphFact(
            subject="retinol",
            predicate="CONFLICTS_WITH",
            object="AHA",
            properties={"reason": "irritation risk"},
            confidence=0.95,
        )
    """
    subject: str
    predicate: str
    object: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_natural_language(self) -> str:
        """Convert fact to a natural language statement."""
        props_str = ""
        if self.properties.get("reason"):
            props_str = f" (reason: {self.properties['reason']})"
        elif self.properties.get("evidence_grade"):
            props_str = f" (evidence: {self.properties['evidence_grade']})"
        return f"{self.subject} {self.predicate.lower().replace('_', ' ')} {self.object}{props_str}"


class GraphRetriever:
    """
    Retrieves structured facts from the skincare knowledge graph (Neo4j).

    STUB IMPLEMENTATION: All methods raise NotImplementedError until Neo4j
    is configured. This allows the rest of the pipeline to import this module
    safely without requiring Neo4j at startup.

    To enable: set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env and
    run scripts/seed_graph.py to populate the database.

    Full documentation: docs/GRAPHRAG_DESIGN.md
    """

    def __init__(
        self,
        neo4j_uri: str = "",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        enabled: bool = False,
    ):
        """
        Args:
            neo4j_uri: Bolt URI, e.g. "bolt://localhost:7687"
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            enabled: Must be True to attempt connection; False = safe no-op
        """
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._enabled = enabled
        self._driver = None

        if enabled and neo4j_uri:
            self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Neo4j."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            self._driver.verify_connectivity()
        except ImportError:
            raise NotImplementedError(
                "Neo4j Python driver not installed. "
                "Run: pip install neo4j>=5.0.0\n"
                "Then: python scripts/seed_graph.py"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {self._uri}: {e}\n"
                "Ensure Neo4j is running and credentials are correct.\n"
                "See docs/GRAPHRAG_DESIGN.md for setup instructions."
            )

    def get_ingredient_relations(
        self,
        ingredient_name: str,
        relation_types: list[str] | None = None,
    ) -> list[GraphFact]:
        """
        Get all graph relationships for an ingredient.

        Args:
            ingredient_name: INCI name (case-insensitive)
            relation_types: Filter to these relationship types (e.g. ["CONFLICTS_WITH"])
                           None = all relationships

        Returns:
            List of GraphFact objects

        Raises:
            NotImplementedError: Until Neo4j is configured

        Example query (once enabled):
            MATCH (i:Ingredient {name: 'retinol'})-[r]->(other)
            RETURN type(r), other.name, properties(r)
        """
        if not self._enabled:
            raise NotImplementedError(
                "GraphRetriever is disabled. To enable:\n"
                "1. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env\n"
                "2. Run: python scripts/seed_graph.py\n"
                "3. Initialize: GraphRetriever(enabled=True)\n"
                "See docs/GRAPHRAG_DESIGN.md"
            )
        raise NotImplementedError(
            "get_ingredient_relations() not yet implemented. "
            "Full Neo4j integration is planned for v2.0. "
            "See docs/GRAPHRAG_DESIGN.md for the full schema."
        )

    def get_condition_treatments(
        self,
        condition_name: str,
        min_evidence_grade: str = "B",
    ) -> list[GraphFact]:
        """
        Get ingredients that treat a specific skin condition.

        Args:
            condition_name: Skin condition name (e.g. "acne", "rosacea")
            min_evidence_grade: Minimum evidence level (A/B/C)

        Returns:
            List of GraphFact with predicate="TREATS_CONDITION"

        Raises:
            NotImplementedError: Until Neo4j is configured

        Example query:
            MATCH (i:Ingredient)-[r:TREATS_CONDITION {evidence_grade: 'A'}]->(c:Condition {name: 'acne'})
            RETURN i.name, r.evidence_grade, r.study_reference
        """
        if not self._enabled:
            raise NotImplementedError(
                "GraphRetriever is disabled. See docs/GRAPHRAG_DESIGN.md for setup."
            )
        raise NotImplementedError(
            "get_condition_treatments() not yet implemented. "
            "See docs/GRAPHRAG_DESIGN.md"
        )

    def augment_retrieval_results(
        self,
        rag_results: list,
        profile: dict,
    ) -> list:
        """
        Augment RAG retrieval results with graph facts.

        This is the main integration point between the vector knowledge base
        and the knowledge graph. For each RAG result, additional structured
        facts are fetched from the graph and appended as metadata.

        Args:
            rag_results: List of RetrievalResult objects from RAGRetriever
            profile: Skin profile dict (used to filter relevant graph facts)

        Returns:
            Same rag_results list with .graph_facts attribute populated

        Raises:
            NotImplementedError: Until Neo4j is configured

        Integration point in pipeline:
            retriever = RAGRetriever(...)
            results = retriever.retrieve(query)
            # Once graph is ready:
            graph = GraphRetriever(enabled=True)
            results = graph.augment_retrieval_results(results, profile)
            generator.generate(profile, results)  # richer context
        """
        if not self._enabled:
            raise NotImplementedError(
                "GraphRetriever is disabled. See docs/GRAPHRAG_DESIGN.md for setup."
            )
        raise NotImplementedError(
            "augment_retrieval_results() not yet implemented. "
            "See docs/GRAPHRAG_DESIGN.md for the integration design."
        )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()

    def __del__(self):
        self.close()
