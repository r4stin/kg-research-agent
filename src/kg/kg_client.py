# src/kg/kg_client.py

from typing import Optional

from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.models.evidence import EvidenceResponse, EvidenceItem
from src.utils.dedup_evidence import _question_hash  # reuse helper


class Neo4jClient:
    """
    Minimal Neo4j client for writing evidence into a graph.

    Schema:

    (p:Paper {paper_id, source})
    (c:Claim {text})
    (e:Evidence {text, chunk_index})

    Relationships:
      (p)-[:HAS_CLAIM]->(c)
      (c)-[:SUPPORTED_BY]->(e)
      (e)-[:EVIDENCE_FOR_QUESTION {question}]->(c)
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD

        if not self.uri or not self.user or not self.password:
            raise RuntimeError("Neo4j connection settings are not fully configured.")

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self._driver is not None:
            self._driver.close()

    # ----- Public API -----

    def upsert_evidence_response(self, evidence: EvidenceResponse):
        """
        Write all EvidenceItems for a given question into the graph.
        """
        with self._driver.session() as session:
            for item in evidence.items:
                session.execute_write(self._upsert_evidence_item, evidence.question, item)

    # ----- Internal helpers -----

    @staticmethod
    def _upsert_evidence_item(tx, question: str, item: EvidenceItem):
        q_hash = _question_hash(question)

        tx.run(
            """
            // Upsert Paper node
            MERGE (p:Paper {paper_id: $paper_id})
              ON CREATE SET p.source = $source

            // Upsert Claim node (still deduped by text)
            MERGE (c:Claim {text: $claim})

            // Upsert Evidence node with strict identity:
            // one node per (paper, chunk, question_hash)
            MERGE (e:Evidence {
              paper_id: $paper_id,
              chunk_index: $chunk_index,
              question_hash: $question_hash
            })
              ON CREATE SET e.text = $evidence_sentence

            // Relationships
            MERGE (p)-[:HAS_CLAIM]->(c)
            MERGE (c)-[:SUPPORTED_BY]->(e)
            MERGE (e)-[:EVIDENCE_FOR_QUESTION {question: $question}]->(c)
            """,
            paper_id=item.paper_id,
            source=item.source,
            claim=item.claim,
            evidence_sentence=item.evidence_sentence,
            chunk_index=item.chunk_index,
            question=question,
            question_hash=q_hash,
        )
