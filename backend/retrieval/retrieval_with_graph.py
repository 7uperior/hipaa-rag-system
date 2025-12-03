"""
Graph-Enhanced Retrieval (Graph RAG).
Combines Postgres Vector Search with Neo4j Graph Traversal.
"""
import os
from openai import OpenAI
from langchain_neo4j import Neo4jGraph

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

graph = Neo4jGraph(
    url="bolt://neo4j:7687",
    username="neo4j",
    password="password123"
)


async def graph_vector_search(query: str, top_k: int = 10):
    """Search Neo4j vector index for similar HIPAA sections."""
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    res = graph.query(
        """
        CALL db.index.vector.queryNodes(
            'section_embedding_idx',
            $top_k,
            $embedding
        )
        YIELD node, score
        RETURN node.section AS section,
               node.content AS content,
               score
        ORDER BY score DESC
        """,
        {"embedding": emb, "top_k": top_k}
    )

    return [
        {
            "section": r["section"],
            "content": r["content"],
            "similarity": float(r["score"]),
            "source": "neo4j_vector"
        }
        for r in res
    ]


async def graph_neighbor_expansion(sections: list[str], depth: int = 1):
    """Expand results through RELATES_TO / REFERENCES edges."""
    res = graph.query(
        """
        MATCH (s:Section)
        WHERE s.section IN $sections
        CALL apoc.path.expand(s, 'RELATES_TO|REFERENCES', null, 1, $depth)
        YIELD path
        WITH DISTINCT last(nodes(path)) AS n
        RETURN n.section AS section, n.content AS content
        """,
        {"sections": sections, "depth": depth}
    )

    return [
        {
            "section": r["section"],
            "content": r["content"],
            "similarity": 0.5, 
            "source": "graph_neighbors"
        }
        for r in res
    ]


async def graph_hybrid_search(query: str, top_k: int = 10):
    """Vector search + graph expansion."""
    base = await graph_vector_search(query, top_k=top_k)

    top_sections = [x["section"] for x in base]

    expanded = await graph_neighbor_expansion(top_sections, depth=1)

    merged = {x["section"]: x for x in base}

    for item in expanded:
        if item["section"] not in merged:
            merged[item["section"]] = item

    results = sorted(
        merged.values(),
        key=lambda x: x.get("similarity", 0),
        reverse=True
    )

    return results[:top_k]
