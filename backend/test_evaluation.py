"""
HIPAA RAG Evaluation Suite (LLM-as-a-Judge)
===========================================

Description:
  This script performs an automated end-to-end evaluation of the HIPAA RAG 
  API. It sends a predefined set of regulatory questions to the local API endpoint, 
  retrieves the generated answers, and uses a secondary LLM (GPT-4o-mini) 
  to act as an impartial judge.

Metrics Evaluated:
  - Accuracy, Completeness, Source Attribution, Relevance, and Clarity.

Output:
  - Console summary of scores.
  - JSON report saved to '/app/evaluation_results_[timestamp].json'.
  - Final verdict (e.g., "Production Ready", "Needs Work").
"""

import os
import json
import requests
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 9 evaluation questions from assignment
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the overall purpose of HIPAA Part 160?",
        "expected_topics": ["statutory basis", "purpose", "administrative simplification"]
    },
    {
        "id": 2,
        "question": "Which part covers data privacy measures?",
        "expected_topics": ["Part 164", "Privacy Rule", "protected health information"]
    },
    {
        "id": 3,
        "question": "What does \"minimum necessary\" mean in HIPAA terminology?",
        "expected_topics": ["minimum amount", "disclosure", "necessary to accomplish"]
    },
    {
        "id": 4,
        "question": "Which entities are specifically regulated under HIPAA?",
        "expected_topics": ["covered entities", "health plans", "healthcare providers", "clearinghouses"]
    },
    {
        "id": 5,
        "question": "What are the potential civil penalties for noncompliance?",
        "expected_topics": ["civil money penalties", "violations", "amounts"]
    },
    {
        "id": 6,
        "question": "Does HIPAA mention encryption best practices?",
        "expected_topics": ["encryption", "security", "safeguards"]
    },
    {
        "id": 7,
        "question": "Can I disclose personal health information to family members?",
        "expected_topics": ["disclosure", "family", "authorization", "opportunity to agree"]
    },
    {
        "id": 8,
        "question": "If a covered entity outsources data processing, which sections apply?",
        "expected_topics": ["business associate", "contract", "requirements"]
    },
    {
        "id": 9,
        "question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
        "expected_topics": ["law enforcement", "disclosure", "164.512"]
    }
]

def ask_system(question: str) -> dict:
    """Send question to HIPAA RAG system."""
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"text": question},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"Error: {response.status_code}", "sources": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}

def evaluate_answer(question: str, answer: str, sources: list, expected_topics: list) -> dict:
    """Evaluate answer quality using LLM-as-a-Judge approach."""
    
    evaluation_prompt = f"""You are an expert evaluator for a HIPAA RAG system.

Question: {question}

System's Answer: {answer}

Sources Cited: {', '.join(sources) if sources else 'None'}

Expected Topics: {', '.join(expected_topics)}

Evaluate the answer on these criteria (score 0-10 for each):

1. ACCURACY: Does the answer correctly reflect HIPAA regulations?
2. COMPLETENESS: Does it cover the expected topics?
3. SOURCE_ATTRIBUTION: Are specific sections (§ XXX.XXX) properly cited?
4. RELEVANCE: Is the answer directly relevant to the question?
5. CLARITY: Is the answer clear and well-structured?

Respond ONLY with valid JSON in this format:
{{
  "accuracy": <score 0-10>,
  "completeness": <score 0-10>,
  "source_attribution": <score 0-10>,
  "relevance": <score 0-10>,
  "clarity": <score 0-10>,
  "total": <sum of all scores>,
  "feedback": "<brief explanation of scores>"
}}

CRITICAL: Output ONLY the JSON object, nothing else."""

    try:
        response = client.chat.completions.create(
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
            "clarity": 0,
            "total": 0,
            "feedback": f"Evaluation failed: {str(e)}"
        }

def run_evaluation():
    """Run full system evaluation with all test questions."""
    
    print("=" * 80)
    print("🧪 HIPAA RAG SYSTEM EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total questions: {len(TEST_QUESTIONS)}\n")
    
    results = []
    total_scores = {
        "accuracy": 0,
        "completeness": 0,
        "source_attribution": 0,
        "relevance": 0,
        "clarity": 0,
        "total": 0
    }
    
    for test in TEST_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"Question {test['id']}: {test['question']}")
        print(f"{'='*80}")
        
        # Query the system
        response = ask_system(test['question'])
        answer = response.get('answer', 'No answer')
        sources = response.get('sources', [])
        
        print(f"\n📝 Answer Preview: {answer[:200]}...")
        print(f"📚 Sources: {', '.join(sources) if sources else 'None'}")
        
        # Evaluate with LLM
        print("\n🤖 Evaluating with GPT-4o-mini...")
        evaluation = evaluate_answer(
            test['question'],
            answer,
            sources,
            test['expected_topics']
        )
        
        # Store results
        result = {
            "question_id": test['id'],
            "question": test['question'],
            "answer": answer,
            "sources": sources,
            "evaluation": evaluation
        }
        results.append(result)
        
        # Update totals
        for key in total_scores.keys():
            total_scores[key] += evaluation.get(key, 0)
        
        # Display scores
        print("\n📊 Scores:")
        print(f"   Accuracy:           {evaluation.get('accuracy', 0)}/10")
        print(f"   Completeness:       {evaluation.get('completeness', 0)}/10")
        print(f"   Source Attribution: {evaluation.get('source_attribution', 0)}/10")
        print(f"   Relevance:          {evaluation.get('relevance', 0)}/10")
        print(f"   Clarity:            {evaluation.get('clarity', 0)}/10")
        print(f"   TOTAL:              {evaluation.get('total', 0)}/50")
        print(f"\n💬 Feedback: {evaluation.get('feedback', 'N/A')}")
    
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
    print(f"   Clarity:            {avg_scores['clarity']:.1f}/10")
    print(f"   OVERALL:            {avg_scores['total']:.1f}/50 ({avg_scores['total']/50*100:.1f}%)")
    
    # Save results
    output_file = f"/app/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(TEST_QUESTIONS),
            "average_scores": avg_scores,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
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
    run_evaluation()