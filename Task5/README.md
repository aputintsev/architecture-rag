Требуемые изменения выполнены в `Task4/rag_bot.py`  
Инструкции по запуску: [Task4/README.md](../Task4/README.md)

Использованы следующие виды защит:
1. На уровне системного промпта + few shot пример. Без few shot небольшая локальная модель типа Qwen3 1.7b, с которой я выполнял тесты, склонна игнорировать инструкции по безопасности в системном промпте.  
   Пример ответа: [скрин](system_prompt_protection_verbose.png)
2. Фильтрация чанков по потенциально вредоносным словам ("superuser password", "admin password", etc).  
   Пример ответа: [скрин](filter_chunks_protection_verbose.png)
3. Фильтрация ответа от LLM на предмет раскрытия чувствительной информации.  
   Пример ответа: [скрин](filter_llm_answer_protection_verbose.png)


Результат работы бота: [bot_output.md](bot_output.md)