# parser.py
import json
import re
from typing import List, Dict, Any

# Импорт ваших моделей
from models import (
    SectionChunk, 
    PartMetadataChunk,
    SubpartMetadataChunk,
    ReservedSectionChunk,
    ReservedSubpartChunk
)


MAX_CHUNK_SIZE = 7500  # Целевой размер чанка (с запасом до 8000)
MIN_CHUNK_SIZE = 3000  # Минимальный размер для объединения
OVERLAP_SIZE = 200     # Размер перекрытия между чанками


def extract_references(text):
    """Извлекает references из текста - только ссылки на секции формата XXX.XXX"""
    references = []
    
    # Паттерны для ссылок на секции (только формат XXX.XXX)
    patterns = [
        r'§\s*(\d+\.\d+)',  # § 164.530
        r'§§\s*(\d+\.\d+)',  # §§ 160.406
        r'\$\\S\s*(\d+\.\d+)',  # $\S 164.530 (LaTeX формат)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        references.extend(matches)
    
    # Удаляем дубликаты, сохраняя порядок
    seen = set()
    unique_refs = []
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    
    return unique_refs


def split_large_section_intelligently(section_data: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    Интеллектуальное разбиение больших секций на подчанки с группировкой.
    
    Стратегия:
    1. Находим все подразделы (a), (b), (c) и т.д.
    2. ГРУППИРУЕМ их в чанки по ~5-7k символов
    3. Только если группа сама слишком большая - разбиваем дальше
    
    Args:
        section_data: Метаданные секции
        text: Полный текст секции
        
    Returns:
        Список подчанков с метаданными
    """
    
    # Паттерн для обнаружения основных подразделов: (a), (b), (c) и т.д.
    subsection_pattern = r'\n\(([a-z])\)\s+\*?([^*\n]+)\*?'
    
    matches = list(re.finditer(subsection_pattern, text))
    
    # Если нет четких подразделов, разбиваем по параграфам с overlap
    if len(matches) < 2:
        return split_by_paragraphs_with_overlap(section_data, text)
    
    # Извлекаем все подразделы с их текстами
    subsections = []
    for i, match in enumerate(matches):
        subsection_letter = match.group(1)
        subsection_title = match.group(2).strip()
        
        start_pos = match.start()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        subsection_text = text[start_pos:end_pos].strip()
        
        subsections.append({
            'letter': subsection_letter,
            'title': subsection_title,
            'text': subsection_text,
            'length': len(subsection_text),
            'position': i
        })
    
    # ГРУППИРУЕМ подразделы в чанки
    chunks = []
    current_group = []
    current_length = 0
    group_index = 0
    
    for subsec in subsections:
        # Если добавление этого подраздела превысит лимит
        if current_length + subsec['length'] > MAX_CHUNK_SIZE:
            # Сохраняем текущую группу (если она не пустая)
            if current_group:
                chunks.append(create_grouped_chunk(section_data, current_group, group_index))
                group_index += 1
                current_group = []
                current_length = 0
            
            # Если один подраздел сам слишком большой - разбиваем его
            if subsec['length'] > MAX_CHUNK_SIZE:
                print(f"   ⚠️  Подраздел ({subsec['letter']}) слишком большой ({subsec['length']} симв.) - разбиваем")
                sub_chunks = split_by_paragraphs_with_overlap(
                    section_data,
                    subsec['text'],
                    subsection_suffix=f"({subsec['letter']})_g{group_index}",
                    max_chunk_size=MAX_CHUNK_SIZE
                )
                chunks.extend(sub_chunks)
                group_index += 1
            else:
                # Начинаем новую группу с этого подраздела
                current_group.append(subsec)
                current_length = subsec['length']
        else:
            # Добавляем к текущей группе
            current_group.append(subsec)
            current_length += subsec['length']
    
    # Сохраняем последнюю группу
    if current_group:
        chunks.append(create_grouped_chunk(section_data, current_group, group_index))
    
    return chunks


def create_grouped_chunk(section_data: Dict[str, Any], subsections: List[Dict], group_index: int) -> Dict[str, Any]:
    """
    Создает чанк из группы подразделов.
    
    Args:
        section_data: Метаданные секции
        subsections: Список подразделов для объединения
        group_index: Индекс группы (для уникальности ID)
        
    Returns:
        Данные чанка
    """
    # Объединяем тексты
    combined_text = "\n\n".join(s['text'] for s in subsections)
    
    # Создаем идентификатор из букв подразделов
    letters = [s['letter'] for s in subsections]
    if len(letters) == 1:
        letter_range = f"({letters[0]})"
    else:
        letter_range = f"({letters[0]}-{letters[-1]})"
    
    # Создаем заголовок
    if len(subsections) == 1:
        title_suffix = f"{letter_range} {subsections[0]['title']}"
    else:
        title_suffix = f"{letter_range} [{len(subsections)} subsections]"
    
    chunk_data = section_data.copy()
    
    # Добавляем group_index для уникальности
    chunk_data["chunk_id"] = f"{section_data['chunk_id']}_sub_g{group_index}_{letters[0]}"
    if len(letters) > 1:
        chunk_data["chunk_id"] += f"_{letters[-1]}"
    
    chunk_data["section_title"] = f"{section_data.get('section_title', '')} {title_suffix}"
    chunk_data["text"] = combined_text
    chunk_data["references"] = extract_references(combined_text)
    chunk_data["is_subchunk"] = True
    chunk_data["parent_section"] = section_data["chunk_id"]
    chunk_data["subsection_marker"] = letter_range
    chunk_data["grouped_subsections"] = letters
    chunk_data["group_index"] = group_index
    
    return chunk_data


def split_by_paragraphs_with_overlap(
    section_data: Dict[str, Any], 
    text: str, 
    subsection_suffix: str = "",
    max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[Dict[str, Any]]:
    """
    Разбивает текст на чанки по параграфам с overlap.
    Используется когда нет четких логических подразделов или подраздел слишком большой.
    """
    
    # Сначала пытаемся разбить по двойным переносам строк (параграфы)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Если получился только один большой "параграф", используем альтернативные точки разбиения
    if len(paragraphs) <= 1:
        # Разбиваем по подпунктам (A), (B), (C) или (i), (ii), (iii) или (1), (2), (3)
        split_pattern = r'\n(?=\([A-Z]\)|\([ivxlcdm]+\)|\(\d+\))'
        segments = re.split(split_pattern, text)
        
        # Если и это не помогло, разбиваем по предложениям
        if len(segments) <= 1 or all(len(s) > max_chunk_size for s in segments):
            sentences = re.split(r'(?<=[.!?])\s+', text)
            paragraphs = sentences
        else:
            paragraphs = segments
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Если добавление параграфа превысит лимит
        if len(current_chunk) + len(para) + 2 > max_chunk_size:
            if current_chunk:
                # Сохраняем текущий чанк
                chunk_data = section_data.copy()
                suffix = f"{subsection_suffix}_p{chunk_index}" if subsection_suffix else f"_p{chunk_index}"
                chunk_data["chunk_id"] = f"{section_data['chunk_id']}{suffix}"
                chunk_data["text"] = current_chunk.strip()
                chunk_data["references"] = extract_references(current_chunk)
                chunk_data["is_subchunk"] = True
                chunk_data["parent_section"] = section_data["chunk_id"]
                chunk_data["chunk_part"] = f"Part {chunk_index + 1}"
                
                chunks.append(chunk_data)
                chunk_index += 1
                
                # Начинаем новый чанк с overlap
                if len(current_chunk) > OVERLAP_SIZE:
                    overlap_text = current_chunk[-OVERLAP_SIZE:]
                    last_period = overlap_text.rfind('. ')
                    if last_period != -1:
                        overlap_text = overlap_text[last_period + 2:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Параграф сам слишком большой - принудительное разбиение
                if len(para) > max_chunk_size:
                    while para:
                        if len(para) <= max_chunk_size:
                            current_chunk = para
                            break
                        
                        split_point = max_chunk_size
                        last_period = para[:split_point].rfind('. ')
                        
                        if last_period != -1 and last_period > max_chunk_size * 0.5:
                            split_point = last_period + 2
                        
                        chunk_piece = para[:split_point].strip()
                        para = para[split_point:].strip()
                        
                        chunk_data = section_data.copy()
                        suffix = f"{subsection_suffix}_p{chunk_index}" if subsection_suffix else f"_p{chunk_index}"
                        chunk_data["chunk_id"] = f"{section_data['chunk_id']}{suffix}"
                        chunk_data["text"] = chunk_piece
                        chunk_data["references"] = extract_references(chunk_piece)
                        chunk_data["is_subchunk"] = True
                        chunk_data["parent_section"] = section_data["chunk_id"]
                        chunk_data["chunk_part"] = f"Part {chunk_index + 1}"
                        
                        chunks.append(chunk_data)
                        chunk_index += 1
                else:
                    current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Сохраняем последний чанк
    if current_chunk:
        chunk_data = section_data.copy()
        suffix = f"{subsection_suffix}_p{chunk_index}" if subsection_suffix else f"_p{chunk_index}"
        chunk_data["chunk_id"] = f"{section_data['chunk_id']}{suffix}"
        chunk_data["text"] = current_chunk.strip()
        chunk_data["references"] = extract_references(current_chunk)
        chunk_data["is_subchunk"] = True
        chunk_data["parent_section"] = section_data["chunk_id"]
        chunk_data["chunk_part"] = f"Part {chunk_index + 1}"
        
        chunks.append(chunk_data)
    
    return chunks


def parse_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_chunks = []

    # --- Переменные состояния (контекст) ---
    current_part = None
    current_part_title = None
    current_subpart = None
    current_subpart_title = None
    
    # --- Буфер для текущей секции ---
    current_section_data = None
    section_text_buffer = []
    section_start_line = None
    
    # Флаги для отслеживания метаданных
    awaiting_part_metadata = False
    awaiting_subpart_metadata = False
    part_metadata_buffer = []
    subpart_metadata_buffer = []
    part_metadata_start_line = None
    subpart_metadata_start_line = None
    
    # Флаг для многострочных заголовков секций
    pending_section_header = None
    pending_section_line = None

    def save_current_chunk():
        """Сохраняет накопленный section буфер в список с проверкой размера."""
        nonlocal current_section_data, section_text_buffer, section_start_line
        
        if current_section_data:
            # Склеиваем накопленные строки текста
            full_text = "\n".join(section_text_buffer).strip()
            
            # Проверяем размер текста
            if len(full_text) > MAX_CHUNK_SIZE:
                print(f"⚠️  Большая секция {current_section_data['chunk_id']}: {len(full_text)} символов - разбиваем с группировкой")
                
                # Интеллектуально разбиваем на подчанки с группировкой
                sub_chunks = split_large_section_intelligently(current_section_data, full_text)
                
                print(f"   ✓ Создано {len(sub_chunks)} подчанков (с группировкой подразделов)")
                
                # Валидируем и добавляем каждый подчанк
                for sub_chunk_data in sub_chunks:
                    try:
                        validated_chunk = SectionChunk(**sub_chunk_data)
                        all_chunks.append(validated_chunk)
                    except Exception as e:
                        cid = sub_chunk_data.get('chunk_id', 'UNKNOWN')
                        print(f"\n{'='*70}")
                        print(f"ОШИБКА ВАЛИДАЦИИ подчанка {cid}")
                        print(f"Строка начала секции: {section_start_line}")
                        print(f"{'='*70}")
                        print(f"Ошибка Pydantic: {e}")
                        print(f"{'='*70}\n")
            else:
                # Обычная секция - сохраняем как есть
                current_section_data["text"] = full_text
                current_section_data["references"] = extract_references(full_text)
                
                try:
                    validated_chunk = SectionChunk(**current_section_data)
                    all_chunks.append(validated_chunk)
                except Exception as e:
                    cid = current_section_data.get('chunk_id', 'UNKNOWN')
                    print(f"\n{'='*70}")
                    print(f"ОШИБКА ВАЛИДАЦИИ SectionChunk {cid}")
                    print(f"Строка начала секции: {section_start_line}")
                    print(f"{'='*70}")
                    print(f"Ошибка Pydantic: {e}")
                    print(f"{'='*70}\n")

            # Очищаем буфер
            current_section_data = None
            section_text_buffer = []
            section_start_line = None

    def save_part_metadata():
        """Сохраняет метаданные PART."""
        nonlocal part_metadata_buffer, awaiting_part_metadata, part_metadata_start_line
        
        if part_metadata_buffer and current_part:
            full_text = "\n".join(part_metadata_buffer).strip()
            
            authority = None
            source = None
            
            for line in part_metadata_buffer:
                clean_line = re.sub(r'^\*{2}|\*{2}$', '', line)
                
                if re.match(r'^(AUTHORITY|Authority):\s*', clean_line):
                    authority = re.sub(r'^(AUTHORITY|Authority):\s*', '', clean_line).strip()
                elif re.match(r'^(SOURCE|Source):\s*', clean_line):
                    source = re.sub(r'^(SOURCE|Source):\s*', '', clean_line).strip()
            
            try:
                chunk = PartMetadataChunk(
                    type="part_metadata",
                    chunk_id=f"{current_part}_metadata",
                    part=current_part,
                    part_title=current_part_title or "",
                    authority=authority,
                    source=source,
                    text=full_text
                )
                all_chunks.append(chunk)
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"ОШИБКА ВАЛИДАЦИИ PartMetadataChunk")
                print(f"Строка: {part_metadata_start_line}")
                print(f"Part: {current_part}")
                print(f"Ошибка: {e}")
                print(f"{'='*70}\n")
        
        part_metadata_buffer = []
        awaiting_part_metadata = False
        part_metadata_start_line = None

    def save_subpart_metadata():
        """Сохраняет метаданные SUBPART."""
        nonlocal subpart_metadata_buffer, awaiting_subpart_metadata, subpart_metadata_start_line
        
        if subpart_metadata_buffer and current_subpart and current_part:
            full_text = "\n".join(subpart_metadata_buffer).strip()
            
            source = None
            for line in subpart_metadata_buffer:
                if re.match(r'^(SOURCE|Source):\s*', line):
                    source = re.sub(r'^(SOURCE|Source):\s*', '', line).strip()
            
            try:
                chunk = SubpartMetadataChunk(
                    type="subpart_metadata",
                    chunk_id=f"{current_part}_{current_subpart}_metadata",
                    part=current_part,
                    subpart=current_subpart,
                    subpart_title=current_subpart_title,
                    source=source,
                    text=full_text
                )
                all_chunks.append(chunk)
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"ОШИБКА ВАЛИДАЦИИ SubpartMetadataChunk")
                print(f"Строка: {subpart_metadata_start_line}")
                print(f"Part: {current_part}, Subpart: {current_subpart}")
                print(f"Ошибка: {e}")
                print(f"{'='*70}\n")
        
        subpart_metadata_buffer = []
        awaiting_subpart_metadata = False
        subpart_metadata_start_line = None

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        
        if not line or line == "---":
            continue

        if pending_section_header:
            if (line.startswith("# PART") or 
                line.startswith("## Subpart") or 
                re.match(r'^##\s+[^\d]+\d', line)):
                sec_num, incomplete_title = pending_section_header
                sec_number = f"§ {sec_num}"
                clean_id = sec_num
                
                final_title = incomplete_title if incomplete_title else "No Title"
                current_section_data = {
                    "type": "section",
                    "part": current_part,
                    "part_title": current_part_title,
                    "subpart": current_subpart,
                    "subpart_title": current_subpart_title,
                    "section": sec_number,
                    "section_title": final_title,
                    "chunk_id": clean_id,
                    "text": "",
                    "references": []
                }
                section_start_line = pending_section_line
                pending_section_header = None
                pending_section_line = None
            else:
                sec_num, incomplete_title = pending_section_header
                completed_title = incomplete_title + " " + line
                
                if completed_title.endswith('.'):
                    sec_number = f"§ {sec_num}"
                    clean_id = sec_num
                    
                    current_section_data = {
                        "type": "section",
                        "part": current_part,
                        "part_title": current_part_title,
                        "subpart": current_subpart,
                        "subpart_title": current_subpart_title,
                        "section": sec_number,
                        "section_title": completed_title,
                        "chunk_id": clean_id,
                        "text": "",
                        "references": []
                    }
                    section_start_line = pending_section_line
                    pending_section_header = None
                    pending_section_line = None
                else:
                    pending_section_header = (sec_num, completed_title)
                continue

        part_match = re.match(r'^#\s+PART\s+(\d+)[^A-Z]*([A-Z].+)$', line)
        if part_match:
            save_current_chunk()
            save_part_metadata()
            save_subpart_metadata()
            
            current_part = part_match.group(1).strip()
            current_part_title = part_match.group(2).strip()
            
            awaiting_part_metadata = True
            part_metadata_start_line = line_num
            continue

        subpart_match = re.match(r'^#{2,3}\s+Subparts?\s+([\w\-]+)(?:\s+\[Reserved\])?\s*(.*)$', line, re.IGNORECASE)
        if subpart_match:
            save_current_chunk()
            save_subpart_metadata()
            
            raw_sub = subpart_match.group(1).strip()
            raw_title = subpart_match.group(2).strip() if subpart_match.group(2) else None
            
            clean_sub = re.sub(r'[^\w\-]', '', raw_sub)
            
            if raw_title:
                raw_title = re.sub(r'^[^\w\s]+', '', raw_title).strip()
            
            is_reserved = "[Reserved]" in line or "[reserved]" in line.lower()
            
            if is_reserved:
                if current_part:
                    try:
                        chunk = ReservedSubpartChunk(
                            type="reserved_subpart",
                            chunk_id=f"{current_part}_{clean_sub}_reserved",
                            part=current_part,
                            subpart=clean_sub,
                            text="[Reserved]"
                        )
                        all_chunks.append(chunk)
                    except Exception as e:
                        print(f"ОШИБКА ReservedSubpartChunk на строке {line_num}: {e}")
                
                current_subpart = clean_sub
                current_subpart_title = None
            else:
                current_subpart = clean_sub
                current_subpart_title = raw_title
                
                awaiting_subpart_metadata = True
                subpart_metadata_start_line = line_num
            
            continue

        if awaiting_part_metadata:
            if re.match(r'^\*{0,2}(AUTHORITY|Authority|SOURCE|Source):\*{0,2}', line):
                clean_line = re.sub(r'^\*{2}|\*{2}$', '', line)
                part_metadata_buffer.append(clean_line)
                continue
            elif re.match(r'^##', line):
                save_part_metadata()
            else:
                if part_metadata_buffer:
                    part_metadata_buffer.append(line)
                    continue
                else:
                    awaiting_part_metadata = False

        if awaiting_subpart_metadata:
            if re.match(r'^(SOURCE|Source):', line):
                subpart_metadata_buffer.append(line)
                continue
            elif re.match(r'^##', line):
                save_subpart_metadata()
            else:
                if subpart_metadata_buffer:
                    subpart_metadata_buffer.append(line)
                    continue
                else:
                    awaiting_subpart_metadata = False

        section_match = re.match(r'^##\s+[^\d]+([\d\.]+)(?:\s+\[Reserved\])?(?:\s+(.+))?$', line, re.IGNORECASE)
        if section_match:
            save_current_chunk()

            sec_num = section_match.group(1).strip()
            sec_title = section_match.group(2)
            
            sec_number = f"§ {sec_num}"
            clean_id = sec_num
            
            is_reserved = "[Reserved]" in line or "[reserved]" in line.lower()
            
            if is_reserved or (sec_title is None) or (sec_title and sec_title.strip() in ["", "[Reserved]", "Reserved]"]):
                if current_part and current_subpart:
                    try:
                        chunk = ReservedSectionChunk(
                            type="reserved_section",
                            chunk_id=f"{clean_id}_reserved",
                            part=current_part,
                            subpart=current_subpart,
                            section=sec_number,
                            text="[Reserved]"
                        )
                        all_chunks.append(chunk)
                    except Exception as e:
                        print(f"ОШИБКА ReservedSectionChunk на строке {line_num}: {e}")
            else:
                final_title = sec_title.strip() if sec_title else ""
                
                if final_title and not final_title.endswith('.'):
                    pending_section_header = (sec_num, final_title)
                    pending_section_line = line_num
                else:
                    if not final_title:
                        final_title = "No Title"
                    
                    current_section_data = {
                        "type": "section",
                        "part": current_part,
                        "part_title": current_part_title,
                        "subpart": current_subpart,
                        "subpart_title": current_subpart_title,
                        "section": sec_number,
                        "section_title": final_title,
                        "chunk_id": clean_id,
                        "text": "",
                        "references": []
                    }
                    section_start_line = line_num
            continue

        if line.startswith("#") and not line.startswith("##") and not line.startswith("# PART"):
            continue

        if current_section_data is not None:
            section_text_buffer.append(line)

    if pending_section_header:
        sec_num, incomplete_title = pending_section_header
        sec_number = f"§ {sec_num}"
        clean_id = sec_num
        
        final_title = incomplete_title if incomplete_title else "No Title"
        current_section_data = {
            "type": "section",
            "part": current_part,
            "part_title": current_part_title,
            "subpart": current_subpart,
            "subpart_title": current_subpart_title,
            "section": sec_number,
            "section_title": final_title,
            "chunk_id": clean_id,
            "text": "",
            "references": []
        }
        section_start_line = pending_section_line
    
    save_current_chunk()
    save_part_metadata()
    save_subpart_metadata()

    return all_chunks

# Запуск
if __name__ == "__main__":
    final_chunks = parse_text_file("data/hipaa_linear_text.txt")
    
    # Статистика
    type_counts = {}
    subchunk_count = 0
    max_length = 0
    
    for chunk in final_chunks:
        chunk_type = chunk.type
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        if hasattr(chunk, 'text') and chunk.text:
            text_len = len(chunk.text)
            max_length = max(max_length, text_len)
            
            if text_len > MAX_CHUNK_SIZE + 500:
                print(f"⚠️  ВНИМАНИЕ: Чанк {chunk.chunk_id} слишком большой: {text_len} символов")
        
        if hasattr(chunk, 'is_subchunk') and chunk.is_subchunk:
            subchunk_count += 1
    
    print(f"\n{'='*70}")
    print(f"ИТОГО: Обработано чанков: {len(final_chunks)}")
    print(f"  Из них подчанков: {subchunk_count}")
    print(f"  Максимальная длина: {max_length} символов")
    print(f"\nРаспределение по типам:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count}")
    print(f"{'='*70}\n")

    json_output = [chunk.model_dump(exclude_none=True) for chunk in final_chunks]
    
    with open("hipaa_data.json.json", "w", encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"Результат сохранён в hipaa_data.json.json")