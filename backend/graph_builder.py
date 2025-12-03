# old_backend/build_graph_old.py

"""
Build HIPAA Knowledge Graph using LangChain Neo4j integration.
"""
import json
import re
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
import os
from typing import List, Set

# Configuration
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class HIPAAGraphBuilder:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        self.sections = []
    
    def load_sections(self):
        print("📖 Loading HIPAA sections...")

        with open('/app/data/hipaa_data.json', 'r') as f:
            raw = json.load(f)

        print(f"📄 Loaded raw {len(raw)} sections")

        # ---- Strategy B: MERGE DUPLICATES ----
        merged = {}

        for item in raw:
            sec = item["section"]
            content = item["content"].strip()

            if sec not in merged:
                merged[sec] = content
            else:
                # # Option 1: take the longer one
                # if len(content) > len(merged[sec]):
                #     merged[sec] = content

                # Option 2: full merge:
                merged[sec] += "\n\n" + content

        # Convert back to list of dicts
        self.sections = [
            {"section": sec, "content": merged[sec]}
            for sec in merged
        ]

        print(f"✅ After merging: {len(self.sections)} unique sections")

    
    def extract_references(self, content: str) -> Set[str]:
        """Extract section references from content."""
        patterns = [
            r'§\s*(\d+\.\d+)',
            r'section\s+(\d+\.\d+)',
            r'see\s+§?\s*(\d+\.\d+)',
        ]
        references = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.update(matches)
        return references
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract key topics."""
        content_lower = content.lower()
        topics = []
        
        topic_keywords = {
            'privacy': ['privacy', 'confidential'],
            'security': ['security', 'safeguard'],
            'disclosure': ['disclose', 'disclosure'],
            'enforcement': ['enforcement', 'penalty'],
            'law_enforcement': ['law enforcement'],
            'business_associate': ['business associate'],
            'encryption': ['encrypt'],
            'breach': ['breach'],
            'authorization': ['authorization'],
            'minimum_necessary': ['minimum necessary'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def clear_graph(self):
        """Clear existing graph."""
        print("🗑️  Clearing existing graph...")
        self.graph.query("MATCH (n) DETACH DELETE n")
        print("✅ Graph cleared")
    
    def build_graph_structure(self):
        """Build graph structure with nodes and relationships."""
        print("🔨 Building graph structure...")
        
        # Create constraint
        self.graph.query("""
            CREATE CONSTRAINT section_id IF NOT EXISTS
            FOR (s:Section) REQUIRE s.id IS UNIQUE
        """)
        
        # Create section nodes
        for i, item in enumerate(self.sections):
            section_id = item['section']
            content = item['content']
            part = section_id.split('.')[0]
            topics = self.extract_topics(content)
            references = self.extract_references(content)
            
            # Create node
            self.graph.query("""
                CREATE (s:Section {
                    id: $id,
                    part: $part,
                    content: $content,
                    topics: $topics
                })
            """, {"id": section_id, "part": part, "content": content[:3000], "topics": topics})
            
            # Create references
            for ref in references:
                self.graph.query("""
                    MATCH (s1:Section {id: $from})
                    MATCH (s2:Section {id: $to})
                    MERGE (s1)-[:REFERENCES]->(s2)
                """, {"from": section_id, "to": ref})
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.sections)} sections...")
        
        # Create topic relationships
        print("🔗 Creating topic relationships...")
        self.graph.query("""
            MATCH (s1:Section), (s2:Section)
            WHERE s1.id < s2.id
              AND size(s1.topics) > 0
              AND any(topic IN s1.topics WHERE topic IN s2.topics)
            MERGE (s1)-[:RELATES_TO]->(s2)
        """)
        
        print("✅ Graph structure built")
    
    def create_vector_index(self):
        """Create vector index using Neo4jVector."""
        print("🧮 Creating vector index...")
        
        # Prepare documents for vector store
        texts = [item['content'] for item in self.sections]
        metadatas = [
            {
                "section": item['section'],
                "part": item['section'].split('.')[0]
            }
            for item in self.sections
        ]
        
        # Create vector store (this will add embeddings to existing nodes)
        Neo4jVector.from_texts(
            texts=texts,
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="hipaa_vector_index",
            node_label="Section",
            text_node_property="content",
            embedding_node_property="embedding",
            metadatas=metadatas,
            search_type="hybrid"  # Enables both vector and keyword search
        )
        
        print("✅ Vector index created")
    
    def print_statistics(self):
        """Print graph statistics."""
        print("\n" + "="*60)
        print("📊 GRAPH STATISTICS")
        print("="*60)
        
        result = self.graph.query("""
            MATCH (s:Section)
            RETURN count(s) as sections
        """)
        print(f"\n📈 Nodes: {result[0]['sections']} sections")
        
        result = self.graph.query("""
            MATCH ()-[r:REFERENCES]->()
            RETURN count(r) as count
        """)
        refs = result[0]['count']
        
        result = self.graph.query("""
            MATCH ()-[r:RELATES_TO]->()
            RETURN count(r) as count
        """)
        relates = result[0]['count']
        
        print("\n🔗 Relationships:")
        print(f"   REFERENCES: {refs}")
        print(f"   RELATES_TO: {relates}")
        print(f"   TOTAL: {refs + relates}")
        
        result = self.graph.query("""
            MATCH (s:Section)<-[:REFERENCES]-(other)
            WITH s.id as section, count(other) as refs
            ORDER BY refs DESC LIMIT 5
            RETURN section, refs
        """)
        
        print("\n🔝 Most Referenced:")
        for record in result:
            print(f"   § {record['section']}: {record['refs']} refs")
        
        print("="*60 + "\n")
    
    def build(self):
        """Build complete knowledge graph."""
        print("\n" + "="*60)
        print("🚀 BUILDING HIPAA KNOWLEDGE GRAPH (LangChain)")
        print("="*60 + "\n")
        
        self.load_sections()
        self.clear_graph()
        self.build_graph_structure()
        self.create_vector_index()
        self.print_statistics()
        
        print("✅ Graph built successfully!\n")
        print("🌐 Neo4j Browser: http://localhost:7474")
        print("   Username: neo4j")
        print("   Password: password123\n")

def main():
    import time
    start = time.time()
    
    builder = HIPAAGraphBuilder()
    builder.build()
    
    print(f"⏱️  Total time: {time.time() - start:.1f}s\n")

if __name__ == "__main__":
    main()