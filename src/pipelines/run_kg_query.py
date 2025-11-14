# src/pipelines/run_kg_query.py

from src.kg.kg_client import Neo4jClient


def run_query(question: str):
    """
    Query Neo4j for all claims and evidence that were extracted
    for a given research question (substring match, case-insensitive).
    """

    question = question.strip()
    if not question:
        print("Please enter a non-empty question substring to search for.")
        return []

    client = Neo4jClient()

    cypher = """
    MATCH (e:Evidence)-[r:EVIDENCE_FOR_QUESTION]->(c:Claim)
    MATCH (p:Paper)-[:HAS_CLAIM]->(c)
    WHERE toLower(r.question) CONTAINS toLower($question)
    RETURN p.paper_id AS paper_id,
           p.source AS source,
           c.text AS claim,
           e.text AS evidence,
           e.chunk_index AS chunk_index,
           r.question AS matched_question
    ORDER BY paper_id, chunk_index
    """

    with client._driver.session() as session:
        results = session.run(cypher, question=question).data()

    client.close()
    return results


def print_results(results):
    if not results:
        print("No results found in the KG.")
        return

    print("\n=== KG QUERY RESULTS ===")
    for i, r in enumerate(results, start=1):
        print(f"\n[Result {i}]")
        print(f"Paper ID     : {r['paper_id']}")
        print(f"Source       : {r['source']}")
        print(f"Claim        : {r['claim']}")
        print(f"Evidence     : {r['evidence']}")
        print(f"Chunk Index  : {r['chunk_index']}")
        print(f"Question Tag : {r['matched_question']}")


def main():
    question = input("Enter a question to search the KG: ")
    results = run_query(question)
    print_results(results)


if __name__ == "__main__":
    main()
