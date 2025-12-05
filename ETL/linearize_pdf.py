"""
HIPAA 3-Column Linearizer
===============================================
"""

import pdfplumber
import re
from pathlib import Path

def is_toc_page(text):
    """Определяет страницу оглавления."""
    if not text: return False
    return text.count('....') > 10 or "Contents" in text[:200]

def clean_specific_artifacts(text):
    """
    Удаляет конкретные фразы-паразиты, которые мешают RAG.
    """
    # 1. Удаляем основной заголовок (в разных вариантах переноса строк)
    # Флаг re.DOTALL не нужен, так как мы хотим матчить конкретные фразы
    
    # Вариант в одну строку
    text = text.replace("HIPAA Administrative Simplification Regulation Text March 2013", "")
    
    # Вариант с переносом строки (как на скриншотах)
    text = text.replace("HIPAA Administrative Simplification Regulation Text\nMarch 2013", "")
    
    # Удаляем "Page X" или просто одиночные цифры, которые остались от номеров страниц
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    return text

def extract_clean_text_from_page(page):
    """
    Режет страницу на 3 колонки и читает их по очереди.
    """
    width = page.width
    height = page.height
    
    # Отступы (чтобы срезать лишнее сверху и снизу)
    top_margin = 60
    bottom_margin = 50
    
    # Ширина одной колонки
    col_width = width / 3
    
    # Определяем зоны колонок (слева, сверху, справа, снизу)
    # Добавляем маленькие отступы (padding) по бокам, чтобы не захватить соседей
    padding = 2 
    
    col1_bbox = (0 + padding, top_margin, col_width - padding, height - bottom_margin)
    col2_bbox = (col_width + padding, top_margin, col_width * 2 - padding, height - bottom_margin)
    col3_bbox = (col_width * 2 + padding, top_margin, width - padding, height - bottom_margin)
    
    page_text = []
    
    for bbox in [col1_bbox, col2_bbox, col3_bbox]:
        try:
            col_crop = page.crop(bbox)
            # x_tolerance=1: склеивать буквы, если они рядом
            # y_tolerance=3: склеивать строки, если они рядом (параграфы)
            text = col_crop.extract_text(x_tolerance=1, y_tolerance=3)
            if text:
                page_text.append(text)
        except ValueError:
            pass 

    return "\n\n".join(page_text)

def process_pdf(pdf_path, output_path):
    print(f"📖 Processing 3-Column PDF: {pdf_path}")
    
    full_doc_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            # 1. ПРОПУСКАЕМ ОБЛОЖКУ (Страница 1)
            # Она ломает логику колонок и создает мусор "U.S. De..."
            if i == 0:
                print(f"   ... skipping Title Page (page 1)")
                continue
                
            if (i + 1) % 20 == 0:
                print(f"   ... processing page {i + 1}/{total}")
            
            # Проверка на оглавление
            raw_text = page.extract_text() or ""
            if is_toc_page(raw_text):
                continue
                
            # Извлекаем текст по колонкам
            col_text = extract_clean_text_from_page(page)
            
            # Чистим от заголовков ПРЯМО СЕЙЧАС
            col_text = clean_specific_artifacts(col_text)
            
            full_doc_text += col_text + "\n"

    # Финальная зачистка
    print("🧹 Final cleaning...")
    full_doc_text = full_doc_text.replace('\xa0', ' ')
    # Убираем множественные переносы строк (больше 2-х)
    full_doc_text = re.sub(r'\n{3,}', '\n\n', full_doc_text)
    
    # Сохраняем
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_doc_text)
        
    print(f"✅ Saved clean text to: {output_file.absolute()}")

if __name__ == "__main__":
    process_pdf('data/hipaa_combined.pdf', 'EDA/hipaa_linear_text.txt')