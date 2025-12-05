#!/usr/bin/env python3
"""
HIPAA PDF Link Extractor - FULL DOCUMENT SCAN
==============================================

Сканирует ВСЕ страницы PDF и извлекает внутренние ссылки.

Output files (в папке EDA/):
    - section_to_page_map.json: mapping секций на страницы
    - all_links.json: полная информация о всех ссылках
    - link_extraction_stats.json: статистика
"""

import PyPDF2
import re
import json
import sys
from pathlib import Path


def build_page_id_mapping(reader):
    """
    Строит mapping: page_object_id → page_number
    """
    page_id_to_num = {}
    
    for page_num, page in enumerate(reader.pages):
        # Получаем indirect object
        if hasattr(page, 'indirect_reference'):
            page_id = page.indirect_reference.idnum
            page_id_to_num[page_id] = page_num + 1  # 1-indexed
        
        # Также пробуем через get_object
        page_obj = page
        if hasattr(page_obj, 'indirect_ref'):
            page_id_to_num[page_obj.indirect_ref.idnum] = page_num + 1
    
    return page_id_to_num


def extract_all_links(pdf_path):
    """
    Извлекает все внутренние ссылки из ВСЕХ страниц PDF.
    """
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        total_pages = len(reader.pages)
        print(f"📄 PDF: {Path(pdf_path).name}")
        print(f"   Total pages: {total_pages}")
        
        # Строим mapping page_id → page_number
        print(f"\n🗺️  Building page ID mapping...")
        page_id_to_num = build_page_id_mapping(reader)
        print(f"   Mapped {len(page_id_to_num)} pages")
        
        links = []
        section_to_page = {}
        
        # Сканируем ВСЕ страницы
        print(f"\n🔗 Scanning ALL {total_pages} pages for links...")
        
        pages_with_links = 0
        
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            
            if '/Annots' not in page:
                continue
            
            pages_with_links += 1
            annots = page['/Annots']
            page_link_count = 0
            
            # Прогресс каждые 10 страниц
            if (page_num + 1) % 10 == 0:
                print(f"   ... processing page {page_num + 1}/{total_pages}")
            
            for annot in annots:
                try:
                    annot_obj = annot.get_object()
                    
                    # Проверяем что это ссылка
                    if annot_obj.get('/Subtype') != '/Link':
                        continue
                    
                    # Извлекаем destination
                    dest = None
                    dest_raw = None
                    
                    if '/Dest' in annot_obj:
                        dest = annot_obj['/Dest']
                        dest_raw = str(dest)
                    elif '/A' in annot_obj:
                        action = annot_obj['/A']
                        if hasattr(action, 'get_object'):
                            action = action.get_object()
                        if '/D' in action:
                            dest = action['/D']
                            dest_raw = str(dest)
                    
                    # Обрабатываем destination
                    dest_page = None
                    page_id_used = None
                    
                    if dest and isinstance(dest, list) and len(dest) > 0:
                        dest_page_ref = dest[0]
                        
                        # КРИТИЧНО: извлекаем idnum из indirect reference
                        if hasattr(dest_page_ref, 'idnum'):
                            # Это IndirectObject - используем idnum
                            page_id = dest_page_ref.idnum
                            page_id_used = page_id
                            dest_page = page_id_to_num.get(page_id)
                        
                        elif hasattr(dest_page_ref, 'indirect_reference'):
                            page_id = dest_page_ref.indirect_reference.idnum
                            page_id_used = page_id
                            dest_page = page_id_to_num.get(page_id)
                        
                        elif isinstance(dest_page_ref, int):
                            # Прямой номер страницы (редко)
                            dest_page = dest_page_ref + 1
                        
                        else:
                            # Последняя попытка - через get_object
                            try:
                                dest_page_obj = dest_page_ref.get_object()
                                dest_page_num = reader.pages.index(dest_page_obj)
                                dest_page = dest_page_num + 1
                            except:
                                pass
                    
                    if dest_page:
                        page_link_count += 1
                    
                    # Извлекаем текст ссылки (номер секции)
                    link_text = None
                    section_id = None
                    
                    # Ищем § X.Y паттерн на этой странице
                    section_matches = re.findall(
                        r'§\s*(\d+\.\d+(?:\([a-z0-9]+\))*)', 
                        page_text
                    )
                    
                    if section_matches and dest_page:
                        # В TOC обычно много ссылок подряд
                        # Берем соответствующую по порядку
                        if page_link_count <= len(section_matches):
                            section_id = section_matches[page_link_count - 1]
                            link_text = f"§ {section_id}"
                            
                            # Сохраняем mapping (первое вхождение)
                            if section_id not in section_to_page:
                                section_to_page[section_id] = dest_page
                    
                    links.append({
                        'source_page': page_num + 1,
                        'dest_page': dest_page,
                        'section': section_id,
                        'link_text': link_text,
                        'dest_raw': dest_raw,
                        'page_id': page_id_used,
                        'resolved': dest_page is not None
                    })
                
                except Exception as e:
                    # Сохраняем информацию об ошибке
                    links.append({
                        'source_page': page_num + 1,
                        'error': str(e),
                        'resolved': False
                    })
        
        print(f"   Found links on {pages_with_links} pages")
        
        return links, section_to_page


def print_statistics(links, section_to_page):
    """Печатает статистику извлечения."""
    
    print(f"\n{'='*70}")
    print(f"📊 EXTRACTION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nTotal links found: {len(links)}")
    
    resolved = [l for l in links if l.get('resolved', False)]
    print(f"Resolved links: {len(resolved)} ({len(resolved)/len(links)*100:.1f}%)")
    
    section_links = [l for l in links if l.get('section') is not None]
    print(f"Section links: {len(section_links)}")
    
    print(f"\nUnique sections mapped: {len(section_to_page)}")
    
    # Группируем по частям HIPAA
    part_160 = [s for s in section_to_page.keys() if s.startswith('160.')]
    part_162 = [s for s in section_to_page.keys() if s.startswith('162.')]
    part_164 = [s for s in section_to_page.keys() if s.startswith('164.')]
    
    print(f"\nSections by HIPAA part:")
    print(f"   Part 160: {len(part_160)} sections")
    print(f"   Part 162: {len(part_162)} sections")
    print(f"   Part 164: {len(part_164)} sections")
    
    if resolved:
        print(f"\n✅ Sample resolved links:")
        for link in resolved[:20]:
            text = link.get('link_text', 'No text')
            print(f"   Page {link['source_page']:2d} → Page {link['dest_page']:3d}  ({text})")
    
    if section_to_page:
        print(f"\n🗺️  Section to Page mapping (sample):")
        for i, (section, page) in enumerate(list(section_to_page.items())[:30]):
            print(f"   § {section:20s} → Page {page:3d}")
        
        # Проверяем важные секции
        test_sections = ['160.101', '160.514', '162.1102', '164.502', '164.512']
        print(f"\n🔍 Testing key sections:")
        for section in test_sections:
            if section in section_to_page:
                print(f"   ✅ § {section} → Page {section_to_page[section]}")
            else:
                print(f"   ❌ § {section} not found")


def save_results(links, section_to_page, output_dir='EDA'):
    """Сохраняет результаты в JSON файлы в указанной папке."""
    
    # Создаем папку EDA если её нет
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving files to {output_path.absolute()}/")
    
    # 1. Сохраняем section mapping
    map_path = output_path / 'section_to_page_map.json'
    with open(map_path, 'w') as f:
        json.dump(section_to_page, f, indent=2, sort_keys=True)
    
    print(f"   📄 section_to_page_map.json")
    print(f"      Section → Page mapping: {len(section_to_page)} entries")
    
    # 2. Сохраняем все ссылки
    links_path = output_path / 'all_links.json'
    with open(links_path, 'w') as f:
        json.dump(links, f, indent=2)
    
    print(f"   📄 all_links.json")
    print(f"      All links data: {len(links)} entries")
    
    # 3. Сохраняем детальную статистику
    stats = {
        'total_links': len(links),
        'resolved_links': len([l for l in links if l.get('resolved', False)]),
        'section_links': len([l for l in links if l.get('section')]),
        'unique_sections': len(section_to_page),
        'sections_by_part': {
            'part_160': sorted([s for s in section_to_page.keys() if s.startswith('160.')]),
            'part_162': sorted([s for s in section_to_page.keys() if s.startswith('162.')]),
            'part_164': sorted([s for s in section_to_page.keys() if s.startswith('164.')])
        },
        'all_sections_sorted': sorted(section_to_page.keys())
    }
    
    stats_path = output_path / 'link_extraction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   📄 link_extraction_stats.json")
    print(f"      Statistics summary")
    
    return map_path, links_path, stats_path


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("Usage: python extract_section_links.py <path_to_pdf>")
        print("\nExample:")
        print("  python extract_section_links.py data/hipaa_combined.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print("="*70)
    print("🔍 HIPAA PDF Link Extractor - FULL DOCUMENT SCAN")
    print("="*70)
    
    # Извлекаем ссылки
    links, section_to_page = extract_all_links(pdf_path)
    
    # Показываем статистику
    print_statistics(links, section_to_page)
    
    # Сохраняем результаты в папку EDA
    save_results(links, section_to_page, output_dir='EDA')
    
    if section_to_page:
        print(f"\n✅ Success! Extracted {len(section_to_page)} section mappings.")
    else:
        print(f"\n⚠️  Warning: No section mappings found!")
        print(f"   Check EDA/all_links.json for debugging.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()