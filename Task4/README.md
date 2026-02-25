# Требования

1. Python 3.11+
2. Pip 24.0+
3. Ollama server установлен и запущен
```shell
curl -fsSL https://ollama.com/install.sh | sh
```
4. Установлена и запущена локальная LLM (тестировалось с Qwen3 1.7b). Можно использовать и другую модель, но потребуется внести соответствующие изменения в `rag_bot.py` (константа OLLAMA_MODEL).
```shell
ollama run qwen3:1.7b
```
5. Подготовлено окружение
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Скрипт запуска бота:

Для запуска в интерактивном режиме, просто выполните команду без аргументов:
```shell
python rag_bot.py
```

Для одиночного запроса, укажите вопрос после флага `--query`:
```shell
python rag_bot.py --query '<your question>'
```

