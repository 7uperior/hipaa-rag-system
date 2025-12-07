"""
Evaluation Test Suite
=====================
LLM-as-a-Judge testing framework for the HIPAA RAG system.
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Test configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OPENAI_MODEL = "gpt-4o-mini"

# Test questions with expected topics
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the overall purpose of HIPAA Part 160?",
        "query_type": "explanation",
        "expected_topics": ["administrative simplification", "general provisions", "definitions"]
    },
    {
        "id": 2,
        "question": "Which part covers data privacy measures?",
        "query_type": "reference_list",
        "expected_topics": ["Part 164", "privacy", "security"]
    },
    {
        "id": 3,
        "question": "What does 'minimum necessary' mean in HIPAA terminology?",
        "query_type": "explanation",
        "expected_topics": ["minimum necessary", "use", "disclosure", "protected health information"]
    },
    {
        "id": 4,
        "question": "Which entities are specifically regulated under HIPAA?",
        "query_type": "reference_list",
        "expected_topics": ["covered entity", "health plan", "health care provider", "clearinghouse"]
    },
    {
        "id": 5,
        "question": "What are the potential civil penalties for noncompliance?",
        "query_type": "citation",
        "expected_topics": ["civil money penalty", "violation", "160.404"]
    },
    {
        "id": 6,
        "question": "Does HIPAA mention encryption best practices?",
        "query_type": "explanation",
        "expected_topics": ["encryption", "security", "technical safeguards"]
    },
    {
        "id": 7,
        "question": "Can I disclose personal health information to family members?",
        "query_type": "explanation",
        "expected_topics": ["family", "disclosure", "involvement", "164.510"]
    },
    {
        "id": 8,
        "question": "If a covered entity outsources data processing, which sections apply?",
        "query_type": "reference_list",
        "expected_topics": ["business associate", "contract", "164.504", "164.314"]
    },
    {
        "id": 9,
        "question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
        "query_type": "citation",
        "expected_topics": ["law enforcement", "164.512", "court order", "subpoena"]
    },
    {
        "id": 10,
        "question": "Where is the security rule located?",
        "query_type": "reference_list",
        "expected_topics": ["Part 164", "Subpart C", "security"]
    },
    {
        "id": 11,
        "question": "Give me full contents of 160.103",
        "query_type": "full_text",
        "expected_topics": ["definitions", "160.103"]
    },
    {
        "id": 12,
        "question": "What are the requirements for patient access to their records?",
        "query_type": "explanation",
        "expected_topics": ["access", "164.524", "right", "records"]
    }
]


async def ask_system(client: httpx.AsyncClient, question: str) -> Dict[str, Any]:
    """Send question to the system."""
    try:
        response = await client.post(
            f"{BACKEND_URL}/ask",
            json={"text": question},
            timeout=60.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


async def evaluate_answer(
    question: str,
    answer: str,
    sources: List[str],
    expected_topics: List[str],
    query_type: str
) -> Dict[str, Any]:
    """Evaluate answer using LLM-as-a-Judge."""
    
    client = OpenAI()
    
    prompt = f"""You are evaluating a HIPAA RAG system response.

Question: {question}
Query Type: {query_type}
Expected Topics: {', '.join(expected_topics)}

System Answer:
{answer}

Sources Cited: {', '.join(sources) if sources else 'None'}

Evaluate the response on these criteria (score 1-5):

1. ACCURACY: Does it correctly address the HIPAA question?
2. COMPLETENESS: Does it cover the expected topics?
3. CITATIONS: Are relevant section numbers cited?
4. RELEVANCE: Is the response focused on the question?
5. QUERY_TYPE_FIT: Does it match the expected query type ({query_type})?

Respond in JSON format:
{{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "citations": <1-5>,
    "relevance": <1-5>,
    "query_type_fit": <1-5>,
    "overall": <1-5>,
    "feedback": "<brief explanation>"
}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(result_text)
    except Exception as e:
        return {
            "accuracy": 0,
            "completeness": 0,
            "citations": 0,
            "relevance": 0,
            "query_type_fit": 0,
            "overall": 0,
            "feedback": f"Evaluation error: {e}"
        }


async def run_evaluation():
    """Run full evaluation suite."""
    
    print("="*70)
    print("🧪 HIPAA RAG System Evaluation")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    
    async def process_question(client: httpx.AsyncClient, test: Dict) -> Dict:
        """Process a single test question."""
        response = await ask_system(client, test['question'])
        answer = response.get('answer', 'No answer')
        sources = response.get('sources', [])
        
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
            "answer": answer[:500] + "..." if len(answer) > 500 else answer,
            "sources": sources,
            "evaluation": evaluation
        }
    
    # Process in batches
    BATCH_SIZE = 3
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for batch_start in range(0, len(TEST_QUESTIONS), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(TEST_QUESTIONS))
            batch = TEST_QUESTIONS[batch_start:batch_end]
            
            print(f"\n🔄 Processing questions {batch_start + 1}-{batch_end}...")
            
            tasks = [process_question(client, test) for test in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
    
    # Calculate summary statistics
    total_score = 0
    max_possible = 0
    scores_by_type = {}
    
    for result in results:
        eval_data = result['evaluation']
        overall = eval_data.get('overall', 0)
        total_score += overall
        max_possible += 5
        
        qtype = result['query_type']
        if qtype not in scores_by_type:
            scores_by_type[qtype] = []
        scores_by_type[qtype].append(overall)
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    for result in results:
        print(f"\n📝 Q{result['question_id']}: {result['question'][:60]}...")
        print(f"   Type: {result['query_type']}")
        print(f"   Overall Score: {result['evaluation'].get('overall', 0)}/5")
        print(f"   Sources: {', '.join(result['sources'][:3])}...")
        print(f"   Feedback: {result['evaluation'].get('feedback', 'N/A')[:100]}")
    
    # Summary
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    
    accuracy = (total_score / max_possible * 100) if max_possible > 0 else 0
    print(f"\n🎯 Overall Accuracy: {accuracy:.1f}%")
    print(f"   Total Score: {total_score}/{max_possible}")
    
    print("\n📊 Scores by Query Type:")
    for qtype, scores in scores_by_type.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"   {qtype}: {avg:.2f}/5 ({len(scores)} questions)")
    
    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "total_questions": len(results),
            "results": results,
            "scores_by_type": {k: sum(v)/len(v) for k, v in scores_by_type.items()}
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation())
