"""
HIPAA PDF Parser
================

Description:
  This script extracts text from HIPAA regulatory PDF documents and 
  structures it into a JSON format based on legal section markers 
  (specifically looking for the '§' symbol).

Usage:
  Run the script directly to parse the hardcoded path, or import 
  'parse_hipaa_pdf' into another module.

Output:
  Generates 'hipaa_data.json' containing section numbers and text chunks.
"""

import PyPDF2
import re
import json

def parse_hipaa_pdf(pdf_path):
    """Parses HIPAA PDF"""
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        
        print(f"Reading {len(reader.pages)} pages...")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += text
            if i % 10 == 0:
                print(f"Page {i}...")
    
    pattern = r'§\s*(\d+\.\d+)\s+([^\n]+?)(?=\n|\r)'
    sections = re.findall(pattern, full_text)
    
    print(f"\nFound {len(sections)} sections")
    
    chunks = []
    lines = full_text.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        match = re.match(r'§\s*(\d+\.\d+)', line)
        
        if match:
            if current_section and current_content:
                content = ' '.join(current_content).strip()
                if len(content) > 50:
                    chunks.append({
                        'section': current_section,
                        'content': content
                    })
            
            current_section = match.group(1)
            current_content = [line]
        
        elif current_section:
            current_content.append(line)
    
    if current_section and current_content:
        content = ' '.join(current_content).strip()
        if len(content) > 50:
            chunks.append({
                'section': current_section,
                'content': content
            })
    
    print(f"Created {len(chunks)} chunks")
    
    with open('/app/hipaa_data.json', 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print("Saved to hipaa_data.json")
    return chunks

if __name__ == "__main__":
    chunks = parse_hipaa_pdf('/app/data/hipaa_combined.pdf')
    print("\nSample chunk:")
    if chunks:
        print(f"Section: {chunks[0]['section']}")
        print(f"Content preview: {chunks[0]['content'][:200]}...")