import json
import re
from pathlib import Path

def main():
    txt_path = 'EDA/hipaa_linear_text.txt'
    stats_path = 'EDA/link_extraction_stats.json'
    output_path = 'data/hipaa_rag_chunks.json'

    if not Path(txt_path).exists():
        print("❌ Run 'python EDA/linearize_columns.py' first!")
        return

    print(f"📂 Loading section list...")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        target_sections = stats.get('all_sections_sorted', [])

    print(f"📖 Reading linear text...")
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = []
    # Начинаем поиск с начала файла
    current_search_idx = 0
    
    print(f"🔍 Extracting {len(target_sections)} RAG chunks...")

    for i, section_id in enumerate(target_sections):
        s_esc = re.escape(section_id)
        
        # --- ИЗМЕНЕНИЕ 1: Улучшенный Regex ---
        # (?m) - включает многострочный режим, чтобы ^ работало для каждой строки
        # ^ - начало строки (чтобы не находить "see § 162.510" внутри текста)
        # (?:§|Section) - ищем слово Section или знак §
        # \s+ - ОБЯЗАТЕЛЬНО ожидаем пробел(ы) после знака
        pattern = rf'(?m)^(?:§|Section)\s+{s_esc}(?:\s|\.|,|\(|$)'
        
        match = re.search(pattern, full_text[current_search_idx:])
        
        found_start = -1
        
        if match:
            # Абсолютная позиция
            found_start = current_search_idx + match.start()
            
            # --- ИЗМЕНЕНИЕ 2: Проверка на Оглавление (TOC) ---
            # Если строка слишком короткая или содержит много точек/цифр в конце — это, вероятно, TOC
            line_end = full_text.find('\n', found_start)
            header_line = full_text[found_start:line_end].strip()
            
            # Простая эвристика: если в строке есть "....." или она заканчивается числом (страницей)
            if "..." in header_line or re.search(r'\.\s*\d+$', header_line):
                # Пробуем найти следующее вхождение
                retry_match = re.search(pattern, full_text[found_start + len(header_line):])
                if retry_match:
                    found_start = found_start + len(header_line) + retry_match.start()

            # Ищем конец текущей секции (начало следующей из списка)
            end_pos = -1
            
            # Пытаемся найти начало ЛЮБОЙ следующей секции, чтобы не зависеть только от i+1
            # (на случай если 162.512 пропущена, но есть 162.514)
            if i + 1 < len(target_sections):
                next_id = target_sections[i+1]
                n_esc = re.escape(next_id)
                # Такой же строгий паттерн для следующей секции
                next_pattern = rf'(?m)^(?:§|Section)\s+{n_esc}(?:\s|\.|,|\(|$)'
                
                next_match = re.search(next_pattern, full_text[found_start + 50:]) # +50 байт, чтобы не найти саму себя
                if next_match:
                    end_pos = found_start + 50 + next_match.start()
            
            # Fallback: Если следующая секция не найдена (или это последняя), 
            # ищем просто следующий заголовок вида "§ 1..." как границу
            if end_pos == -1:
                 # Ищем любой следующий параграф, начинающийся с новой строки
                 generic_next = re.search(r'(?m)^(?:§|Section)\s+\d+\.\d+', full_text[found_start + 200:])
                 if generic_next:
                     end_pos = found_start + 200 + generic_next.start()
                 else:
                     end_pos = min(len(full_text), found_start + 20000) # Максимум 20к символов на секцию

            content = full_text[found_start:end_pos].strip()
            
            chunks.append({
                "id": section_id,
                "text": content,
                "metadata": {
                    "source": "hipaa_combined.pdf",
                    "section": section_id,
                    "part": section_id.split('.')[0]
                }
            })
            
            # Обновляем индекс поиска, чтобы следующую секцию искать ПОСЛЕ текущей
            # Важно: ставим курсор чуть дальше начала текущей, но не в самый конец, 
            # на случай если "конец" определен неверно.
            # Но лучше ставить в начало найденного + 1, чтобы сохранить порядок.
            current_search_idx = found_start + 1
            
        else:
            print(f"   ⚠️ Text for § {section_id} not found starting from idx {current_search_idx}")
            # ВАЖНО: Не двигаем current_search_idx, если не нашли. 
            # Возможно, мы пропустили секцию, но следующая (i+1) все еще может быть найдена дальше.

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Created {len(chunks)} RAG-ready chunks")
    print(f"� Saved to: {output_path}")

if __name__ == "__main__":
    main()