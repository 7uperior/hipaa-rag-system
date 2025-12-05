import json
import statistics

def analyze_chunks(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Берем только секции, метаданные обычно короткие
    section_lengths = []
    oversized_chunks = []

    for chunk in chunks:
        if chunk['type'] == 'section':
            # Считаем длину текста + заголовков (так как loader их склеивает)
            # Примерная длина заголовков ~100 символов, но лучше считать честно, 
            # если хотите точности, но для оценки хватит длины текста.
            length = len(chunk['text'])
            section_lengths.append(length)
            
            if length > 8000:
                oversized_chunks.append({
                    "id": chunk['chunk_id'],
                    "len": length,
                    "title": chunk.get('section_title', 'No Title')
                })

    if not section_lengths:
        print("Секции не найдены.")
        return

    print(f"{'='*40}")
    print(f"📊 СТАТИСТИКА ЧАНКОВ (только type='section')")
    print(f"{'='*40}")
    print(f"Всего секций:    {len(section_lengths)}")
    print(f"Минимальная длина: {min(section_lengths)}")
    print(f"Средняя длина:     {int(statistics.mean(section_lengths))}")
    print(f"Медианная длина:   {int(statistics.median(section_lengths))}")
    print(f"Максимальная длина: {max(section_lengths)}")
    print(f"{'-'*40}")
    
    print(f"⚠️ Чанки длиннее 8000 символов: {len(oversized_chunks)}")
    
    if oversized_chunks:
        print("\nСписок 'обрезанных' чанков (Top 10):")
        # Сортируем от самых больших
        oversized_chunks.sort(key=lambda x: x['len'], reverse=True)
        for c in oversized_chunks[:10]:
            print(f" • {c['id']:<10} | {c['len']} симв. | {c['title']}")

if __name__ == "__main__":
    analyze_chunks("ETL/hipaa_chunks_grouped.json")