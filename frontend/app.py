"""
HIPAA Assistant UI
==================

Description:
  This script launches a Gradio web interface for the HIPAA RAG system.
  It acts as a frontend client, sending user queries to the FastAPI backend 
  service and rendering the returned answers and citations in a user-friendly 
  Markdown format.

Dependencies:
  - gradio, requests
  - Requires the backend service running at 'http://backend:8000'.

Usage:
  Run this script to start the web server on port 7860.
  Access via browser at http://localhost:7860.
"""

import gradio as gr
import requests

def ask_hipaa(question):
    """Отправляет вопрос в backend и возвращает ответ"""
    try:
        response = requests.post(
            "http://backend:8000/ask",
            json={"text": question},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            answer = data.get("answer", "No answer provided")
            sources = data.get("sources", [])
            
            result = f"### Answer:\n\n{answer}\n\n"
            
            if sources:
                result += "### 📚 Sources Referenced:\n\n"
                for source in sources:
                    result += f"- {source}\n"
            
            return result
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Make sure the backend service is running."
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="HIPAA Assistant") as demo:
    gr.Markdown(
        """
        # 🏥 HIPAA Regulatory Assistant
        
        Ask questions about HIPAA Parts 160, 162, and 164 regulations.
        The system will provide accurate answers with specific section references.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your HIPAA Question",
                placeholder="e.g., What is the purpose of HIPAA Part 160?",
                lines=4,
                max_lines=10
            )
            
            submit_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Example Questions:
                - What is the overall purpose of HIPAA Part 160?
                - What does "minimum necessary" mean?
                - Can I disclose health information to family members?
                - What are the civil penalties for noncompliance?
                - Cite the regulation regarding law enforcement disclosures
                """
            )
    
    with gr.Row():
        answer_output = gr.Markdown(
            label="Answer",
            value="*Your answer will appear here...*"
        )
    
    submit_btn.click(
        fn=ask_hipaa,
        inputs=question_input,
        outputs=answer_output
    )
    
    question_input.submit(
        fn=ask_hipaa,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )