"""
HIPAA RAG Evaluation Suite (Async) - Updated Version
=====================================================

Fast async evaluation using httpx instead of requests.
Updated to match the new query classification system and chunk structure.
"""

import os
import json
import httpx
import asyncio
from openai import AsyncOpenAI
from datetime import datetime

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 11 evaluation questions covering all query types
TEST_QUESTIONS = [
    # EXPLANATION queries (general questions requiring synthesis)
    {
        "id": 1,
        "question": "What is the overall purpose of HIPAA Part 160?",
        "expected_topics": ["statutory basis", "purpose", "administrative simplification"],
        "query_type": "explanation"
    },
    {
        "id": 2,
        "question": "Which entities are specifically regulated under HIPAA?",
        "expected_topics": ["covered entities", "health plans", "healthcare providers", "clearinghouses"],
        "query_type": "explanation"
    },
    {
        "id": 3,
        "question": "What does \"minimum necessary\" mean in HIPAA terminology?",
        "expected_topics": ["minimum amount", "disclosure", "necessary to accomplish"],
        "query_type": "explanation"
    },
    {
        "id": 4,
        "question": "Can I disclose personal health information to family members?",
        "expected_topics": ["disclosure", "family", "authorization", "opportunity to agree"],
        "query_type": "explanation"
    },
    
    # CITATION queries (specific quotes and exact wording)
    {
        "id": 5,
        "question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
        "expected_topics": ["law enforcement", "disclosure", "164.512", "exact quotes"],
        "query_type": "citation"
    },
    {
        "id": 6,
        "question": "What are the exact words used to define 'protected health information'?",
        "expected_topics": ["PHI", "definition", "164.501", "exact wording"],
        "query_type": "citation"
    },
    
    # REFERENCE_LIST queries (asking for multiple sections)
    {
        "id": 7,
        "question": "Which sections cover security and encryption requirements?",
        "expected_topics": ["security", "encryption", "164.312", "164.314", "list of sections"],
        "query_type": "reference_list"
    },
    {
        "id": 8,
        "question": "Where security rule is located?",
        "expected_topics": ["security rule", "Part 164", "Subpart C", "section references"],
        "query_type": "reference_list"
    },
    
    # FULL_TEXT queries (requesting complete section content)
    {
        "id": 9,
        "question": "Give me full contents of 160.514 part",
        "expected_topics": ["160.514", "complete text", "all subsections", "Other requirements relating to uses and disclosures of protected health information"],
        "query_type": "full_text"
    },
    {
        "id": 10,
        "question": "Show me the complete text of section 164.512",
        "expected_topics": ["164.512", "full text", "uses and disclosures", "authorization not required"],
        "query_type": "full_text"
    },
    
    # MIXED - testing business associate concepts
    {
        "id": 11,
        "question": "If a covered entity outsources data processing, which sections apply?",
        "expected_topics": ["business associate", "contract", "requirements", "164.502", "164.504"],
        "query_type": "reference_list"
    }
]


async def ask_system(http_client: httpx.AsyncClient, question: str) -> dict:
    """Send question to HIPAA RAG system (async)."""
    try:
        response = await http_client.post(
            "http://localhost:8000/ask",
            json={"text": question},
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"Error: {response.status_code}", "sources": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


async def evaluate_answer(question: str, answer: str, sources: list, expected_topics: list, query_type: str) -> dict:
    """Evaluate answer using async OpenAI with query type awareness."""
    
    evaluation_prompt = f"""You are an expert evaluator for a HIPAA RAG system.

Question: {question}
Query Type: {query_type}

System's Answer: {answer}

Sources Cited: {', '.join(sources) if sources else 'None'}

Expected Topics: {', '.join(expected_topics)}

Evaluate the answer on these criteria (score 0-10 for each):

1. ACCURACY: Does the answer correctly reflect HIPAA regulations?
2. COMPLETENESS: Does it cover the expected topics adequately?
3. SOURCE_ATTRIBUTION: Are specific sections (§ XXX.XXX) properly cited?
4. RELEVANCE: Is the answer directly relevant to the question?
5. QUERY_TYPE_MATCH: Does the answer format match the query type?
   - full_text: Should provide complete section text with all subsections
   - citation: Should include exact quotes from regulations
   - explanation: Should synthesize information clearly
   - reference_list: Should list relevant sections with brief descriptions

Respond ONLY with valid JSON in this format:
{{
  "accuracy": <score 0-10>,
  "completeness": <score 0-10>,
  "source_attribution": <score 0-10>,
  "relevance": <score 0-10>,
  "query_type_match": <score 0-10>,
  "total": <sum of all scores>,
  "feedback": "<brief explanation of scores>"
}}

CRITICAL: Output ONLY the JSON object, nothing else."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        evaluation = json.loads(result_text)
        return evaluation
    
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "source_attribution": 0,
            "relevance": 0,
            "query_type_match": 0,
            "total": 0,
            "feedback": f"Evaluation failed: {str(e)}"
        }

async def run_evaluation():
    """Run evaluation with batched parallel processing (3 at a time)."""
    
    print("=" * 80)
    print("🧪 HIPAA RAG SYSTEM EVALUATION (UPDATED VERSION)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total questions: {len(TEST_QUESTIONS)}")
    print(f"Query types tested: full_text, citation, explanation, reference_list\n")
    
    results = []
    total_scores = {
        "accuracy": 0,
        "completeness": 0,
        "source_attribution": 0,
        "relevance": 0,
        "query_type_match": 0,
        "total": 0
    }
    
    async def process_question(http_client: httpx.AsyncClient, test: dict) -> dict:
        """Process single question."""
        # Query system
        response = await ask_system(http_client, test['question'])
        answer = response.get('answer', 'No answer')
        sources = response.get('sources', [])
        
        # Evaluate
        evaluation = await evaluate_answer(
            test['question'],
            answer,
            sources,
            test['expected_topics'],
            test['query_type']
        )
        
        return {
            "question_id": test['id'],
            "question": test['question'],
            "query_type": test['query_type'],
            "answer": answer,
            "sources": sources,
            "evaluation": evaluation
        }
    
    # Process in batches of 3
    BATCH_SIZE = 3
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for batch_start in range(0, len(TEST_QUESTIONS), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(TEST_QUESTIONS))
            batch = TEST_QUESTIONS[batch_start:batch_end]
            
            print(f"🔄 Processing questions {batch_start+1}-{batch_end}...\n")
            
            tasks = [process_question(http_client, test) for test in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
    
    # Display results grouped by query type
    query_type_groups = {}
    for result in results:
        qtype = result['query_type']
        if qtype not in query_type_groups:
            query_type_groups[qtype] = []
        query_type_groups[qtype].append(result)
    
    # Display results
    for query_type in ['full_text', 'citation', 'explanation', 'reference_list']:
        if query_type not in query_type_groups:
            continue
            
        print(f"\n\n{'='*80}")
        print(f"📋 QUERY TYPE: {query_type.upper()}")
        print(f"{'='*80}")
        
        for result in sorted(query_type_groups[query_type], key=lambda x: x['question_id']):
            test_id = result['question_id']
            question = result['question']
            answer = result['answer']
            sources = result['sources']
            evaluation = result['evaluation']
            
            print(f"\n{'-'*80}")
            print(f"Question {test_id}: {question}")
            print(f"{'-'*80}")
            
            print(f"\n📝 Answer Preview: {answer[:300]}...")
            print(f"📚 Sources: {', '.join(sources) if sources else 'None'}")
            
            print("\n📊 Scores:")
            print(f"   Accuracy:           {evaluation.get('accuracy', 0)}/10")
            print(f"   Completeness:       {evaluation.get('completeness', 0)}/10")
            print(f"   Source Attribution: {evaluation.get('source_attribution', 0)}/10")
            print(f"   Relevance:          {evaluation.get('relevance', 0)}/10")
            print(f"   Query Type Match:   {evaluation.get('query_type_match', 0)}/10")
            print(f"   TOTAL:              {evaluation.get('total', 0)}/50")
            print(f"\n💬 Feedback: {evaluation.get('feedback', 'N/A')}")
            
            # Update totals
            for key in total_scores.keys():
                total_scores[key] += evaluation.get(key, 0)
    
    # Final report
    print(f"\n\n{'='*80}")
    print("📈 FINAL RESULTS")
    print(f"{'='*80}")
    
    avg_scores = {k: v / len(TEST_QUESTIONS) for k, v in total_scores.items()}
    
    print("\n🎯 Average Scores:")
    print(f"   Accuracy:           {avg_scores['accuracy']:.1f}/10")
    print(f"   Completeness:       {avg_scores['completeness']:.1f}/10")
    print(f"   Source Attribution: {avg_scores['source_attribution']:.1f}/10")
    print(f"   Relevance:          {avg_scores['relevance']:.1f}/10")
    print(f"   Query Type Match:   {avg_scores['query_type_match']:.1f}/10")
    print(f"   OVERALL:            {avg_scores['total']:.1f}/50 ({avg_scores['total']/50*100:.1f}%)")
    
    # Query type breakdown
    print("\n\n📊 PERFORMANCE BY QUERY TYPE:")
    print(f"{'='*80}")
    
    for query_type, questions in query_type_groups.items():
        type_scores = {
            "accuracy": 0,
            "completeness": 0,
            "source_attribution": 0,
            "relevance": 0,
            "query_type_match": 0,
            "total": 0
        }
        
        for result in questions:
            evaluation = result['evaluation']
            for key in type_scores.keys():
                type_scores[key] += evaluation.get(key, 0)
        
        type_avg = {k: v / len(questions) for k, v in type_scores.items()}
        
        print(f"\n{query_type.upper()} ({len(questions)} questions):")
        print(f"  Average: {type_avg['total']:.1f}/50 ({type_avg['total']/50*100:.1f}%)")
        print(f"  Query Type Match: {type_avg['query_type_match']:.1f}/10")
    
    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(TEST_QUESTIONS),
            "average_scores": avg_scores,
            "query_type_breakdown": {
                qtype: {
                    "count": len(questions),
                    "average_total": sum(r['evaluation'].get('total', 0) for r in questions) / len(questions)
                }
                for qtype, questions in query_type_groups.items()
            },
            "detailed_results": sorted(results, key=lambda x: x['question_id'])
        }, f, indent=2)
    
    print(f"\n\n💾 Detailed results saved to: {output_file}")
    
    # Final verdict
    overall_percent = avg_scores['total'] / 50 * 100
    
    print(f"\n{'='*80}")
    if overall_percent >= 90:
        print("🏆 EXCELLENT! System is production-ready!")
    elif overall_percent >= 75:
        print("✅ GOOD! System meets requirements with minor improvements needed.")
    elif overall_percent >= 60:
        print("⚠️  ACCEPTABLE! System works but needs improvements.")
    else:
        print("❌ NEEDS WORK! System requires significant improvements.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(run_evaluation())