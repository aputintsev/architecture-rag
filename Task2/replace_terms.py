import json
import os
import re

def load_terms_map(terms_file):
    """Загружает словарь замен из JSON-файла."""
    with open(terms_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_safe_pattern(terms):
    """Создает безопасное регулярное выражение для замены целых слов."""
    # Экранируем специальные символы и сортируем по длине
    escaped_terms = [re.escape(term) for term in sorted(terms.keys(), key=len, reverse=True)]

    # Создаем паттерн с границами слов для точного совпадения
    # Учитываем, что некоторые термины могут содержать пробелы или спецсимволы
    pattern = r'(?<!\w)(' + '|'.join(escaped_terms) + r')(?!\w)'
    return re.compile(pattern, re.IGNORECASE)

def replace_terms_in_file_safe(file_path, terms_map):
    """Безопасная замена ключей на значения в содержимом файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Делаем case-insensitive мапу для lookup
        terms_map_ci = {k.casefold(): v for k, v in terms_map.items()}

        # Создаем паттерн для безопасной замены
        pattern = create_safe_pattern(terms_map)

        # Функция замены
        def replace_match(match):
            term = match.group(1)
            key = term.casefold()

            replacement = terms_map_ci.get(key)
            if replacement is None:
                # На всякий случай не падаем, а оставляем как есть
                # (в идеале такого быть не должно, но это спасает прогон)
                return term
            return replacement

        # Выполняем замену
        new_content, count = pattern.subn(replace_match, content)

        # Если были замены, записываем обратно
        if count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Обновлен: {file_path} (заменено: {count})")
            return True, count
        else:
            print(f"Без изменений: {file_path}")
            return False, 0

    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return False, 0

def process_knowledge_base_safe(knowledge_base_dir, terms_file):
    """Обрабатывает все файлы в папке knowledge_base (безопасная версия)."""

    # Загружаем словарь замен
    terms_map = load_terms_map(terms_file)
    print(f"Загружено {len(terms_map)} терминов для замены")

    # Проверяем существование папки
    if not os.path.exists(knowledge_base_dir):
        print(f"Папка {knowledge_base_dir} не существует!")
        return

    # Получаем все файлы с текстовым содержимым
    text_extensions = {'.txt', '.md', '.json', '.xml', '.html', '.csv', '.py', '.js', '.css', '.yaml', '.yml'}

    all_files = []
    for root, dirs, files in os.walk(knowledge_base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions or not ext:  # обрабатываем файлы без расширения как текстовые
                all_files.append(file_path)

    print(f"Найдено {len(all_files)} текстовых файлов для обработки")

    # Обрабатываем каждый файл
    modified_count = 0
    total_replacements = 0

    for file_path in all_files:
        modified, count = replace_terms_in_file_safe(file_path, terms_map)
        if modified:
            modified_count += 1
            total_replacements += count

    print(f"\nОбработка завершена.")
    print(f"Изменено файлов: {modified_count}")
    print(f"Всего замен: {total_replacements}")

if __name__ == "__main__":
    # Путь к папке knowledge_base
    knowledge_base_dir = "knowledge_base"

    # Путь к файлу с терминами
    terms_file = "terms_map.json"

    # Запускаем безопасную версию
    process_knowledge_base_safe(knowledge_base_dir, terms_file)