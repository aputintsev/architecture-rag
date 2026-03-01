# Результаты
Для построения эмбеддингов использована модель all-MiniLM-L6-v2.
Для хранения эмбеддингов используется векторная БД ChromaDB.
В индексе 479 чанков.
Генерация заняла 1 мин 11 с на MacBook Pro M3 36Гб RAM.

# Подготовка окружения
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Запуск
Построение индекса
```shell
python build_index.py
```

Запрос к индексу, выводящий топ-5 чанков
```shell
python build_index.py --query '<your_question>'
``` 