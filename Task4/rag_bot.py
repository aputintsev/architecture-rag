#!/usr/bin/env python3
"""
Task 4 — RAG-бот с техниками промптинга
========================================

Пайплайн:
  1. Пользователь вводит вопрос.
  2. Вопрос преобразуется в эмбеддинг (all-MiniLM-L6-v2).
  3. ChromaDB возвращает top-k релевантных чанков.
  4. Строится промпт: System (CoT) + Few-shot примеры + контекст + вопрос.
  5. Промпт отправляется в Ollama (qwen3:1.7b).
  6. Ответ выводится пользователю.

Запуск:
    python rag_bot.py
    python rag_bot.py --verbose        # показывать найденные чанки
    python rag_bot.py --model qwen3:1.7b
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import httpx
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "Task3" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TIMEOUT = 120.0

TOP_K = 4  # количество извлекаемых чанков

# ---------------------------------------------------------------------------
# System prompt — Chain-of-Thought инструкция
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
/no_think
You are a knowledge base assistant. Your job is to answer questions \
using ONLY the context documents provided in each message.

Always follow this Chain-of-Thought reasoning pattern:
1. Identify what information is needed to answer the question.
2. Find the relevant facts in the provided context documents.
3. Reason through the answer step by step, citing the source.
4. State your final conclusion clearly.

Rules:
- Base your answer exclusively on the provided context. Do not use outside knowledge.
- If the context does not contain enough information, state that explicitly.
- Always show your numbered reasoning steps before the final answer.
- Keep answers concise but complete.\
"""

# ---------------------------------------------------------------------------
# Few-shot примеры — взяты из той же предметной области (KB)
#
# Имена собственных заменены UUID (см. Task2/terms_map.json):
#   Drizzt          → 1e1baf09-7023-446f-9a5e-06bfd0cbde78
#   Guenhwyvar      → e91747f4-88a3-4603-afc6-71d40226724c
#   Menzoberranzan  → 0158a791-a511-46cb-934e-d94ef787df5d
#   Lolth           → dd665bc9-5adb-48aa-a6f0-236bce62b727
#   Drow            → 4ccc52ab-ada3-4c7d-b21e-0192cd9caa40
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "question": (
            "What is the animal companion of 1e1baf09-7023-446f-9a5e-06bfd0cbde78?"
        ),
        "answer": (
            "1. I need to find information about companions of entity "
            "1e1baf09-7023-446f-9a5e-06bfd0cbde78.\n"
            "2. The document states: '1e1baf09-7023-446f-9a5e-06bfd0cbde78 also carried "
            "a unique figurine of wondrous power that summoned his black panther companion "
            "e91747f4-88a3-4603-afc6-71d40226724c.' The figurine was originally owned by "
            "Masoj Hun'ett and was made of onyx.\n"
            "3. Therefore, the animal companion of 1e1baf09-7023-446f-9a5e-06bfd0cbde78 "
            "is e91747f4-88a3-4603-afc6-71d40226724c — a black panther summoned via an "
            "onyx figurine of wondrous power."
        ),
    },
    {
        "question": (
            "Which deity is the primary object of worship in "
            "0158a791-a511-46cb-934e-d94ef787df5d?"
        ),
        "answer": (
            "1. I need to identify the deity worshipped in city "
            "0158a791-a511-46cb-934e-d94ef787df5d.\n"
            "2. According to the documents, the city's buildings were covered with spiders "
            "'in honor of dd665bc9-5adb-48aa-a6f0-236bce62b727' and the city was founded "
            "by 'a powerful priestess of dd665bc9-5adb-48aa-a6f0-236bce62b727'. The "
            "4ccc52ab-ada3-4c7d-b21e-0192cd9caa40 are described as her devoted followers.\n"
            "3. Therefore, the primary deity worshipped in "
            "0158a791-a511-46cb-934e-d94ef787df5d is dd665bc9-5adb-48aa-a6f0-236bce62b727, "
            "known as the Queen of Spiders."
        ),
    },
]


# ---------------------------------------------------------------------------
# Embedding wrapper (идентичен Task 3)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-совместимая обёртка над sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def load_index(embedding_model: SentenceTransformerEmbeddings) -> Chroma:
    """Загрузить существующий Chroma-индекс из Task3."""
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Индекс не найден: {CHROMA_PERSIST_DIR}\n"
            "Сначала выполните Task3/build_index.py"
        )
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model,
        collection_name="knowledge_base",
    )


def retrieve(vector_store: Chroma, query: str, k: int = TOP_K) -> list[dict]:
    """
    Шаг 1 RAG: найти top-k наиболее релевантных чанков для запроса.

    Возвращает список словарей с полями: content, source, title, score.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", "unknown"),
            "score": score,
        }
        for doc, score in results
    ]


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Шаг 2 RAG: сформировать список сообщений для chat-API.

    Структура:
        [system]          — CoT-инструкция
        [user/assistant]  — Few-shot пример 1
        [user/assistant]  — Few-shot пример 2
        [user]            — Контекст из БЗ + вопрос пользователя
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # --- Few-shot примеры ---
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["answer"]})

    # --- Контекст из векторной базы ---
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Document {i} | {chunk['source']} | relevance={chunk['score']:.4f}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # --- Финальное сообщение пользователя ---
    user_message = (
        f"Context documents:\n\n{context}\n\n"
        f"{'─' * 40}\n\n"
        f"Question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


def _strip_think_tags(text: str) -> tuple[str, str]:
    """
    Qwen3 иногда возвращает <think>...</think> блок внутренних размышлений.
    Разделяем его и чистый ответ.

    Возвращает (thinking, answer).
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return thinking, answer
    return "", text.strip()


def generate(messages: list[dict], model: str = OLLAMA_MODEL) -> tuple[str, str]:
    """
    Шаг 3 RAG: отправить промпт в Ollama и вернуть ответ.

    Возвращает (thinking, answer) — thinking может быть пустой строкой.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,   # меньше — фактичнее
            "num_predict": 2048,
        },
    }

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        response = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()

    raw = response.json()["message"]["content"]
    return _strip_think_tags(raw)


def rag_query(
    vector_store: Chroma,
    query: str,
    model: str = OLLAMA_MODEL,
    verbose: bool = False,
) -> str:
    """
    Полный RAG пайплайн для одного вопроса.

    Возвращает строку с ответом модели.
    """
    # Шаг 1: Retrieval
    chunks = retrieve(vector_store, query)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Найдено {len(chunks)} релевантных чанков:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] score={c['score']:.4f}  {c['title']}  ({c['source']})")
        print(f"{'─' * 60}\n")

    # Шаг 2: Build prompt (Few-shot + CoT)
    messages = build_messages(query, chunks)

    # Шаг 3: Generate
    thinking, answer = generate(messages, model=model)

    if verbose and thinking:
        print("[ Внутренние размышления модели (Qwen3 <think>) ]")
        print(thinking)
        print(f"{'─' * 60}\n")

    return answer


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

REPL_HELP = """\
Команды:
  exit / quit  — выход
  verbose      — включить/выключить показ найденных чанков
  help         — эта справка
"""


def repl(vector_store: Chroma, model: str = OLLAMA_MODEL) -> None:
    """Интерактивный консольный интерфейс."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║      RAG-бот · QuantumForge Knowledge Base          ║")
    print(f"║      Модель: {model:<44}║")
    print(f"║      Индекс: {str(CHROMA_PERSIST_DIR):<44}║")
    print("╚" + "═" * 58 + "╝")
    print(REPL_HELP)

    verbose = False

    while True:
        try:
            user_input = input("Вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit", "выход"):
            print("До свидания!")
            break

        if cmd == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON (показываю найденные чанки)' if verbose else 'OFF'}")
            continue

        if cmd == "help":
            print(REPL_HELP)
            continue

        print("\nИщу в базе знаний…\n")

        try:
            answer = rag_query(vector_store, user_input, model=model, verbose=verbose)
            print("Ответ:\n")
            print(answer)
            print()
        except httpx.ConnectError:
            print(
                "Ошибка подключения к Ollama.\n"
                "Убедитесь, что сервер запущен: ollama serve\n"
            )
        except httpx.HTTPStatusError as exc:
            print(f"Ошибка HTTP {exc.response.status_code}: {exc.response.text}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Неожиданная ошибка: {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG-бот с Few-shot и Chain-of-Thought промптингом."
    )
    parser.add_argument(
        "--model", "-m",
        default=OLLAMA_MODEL,
        help=f"Ollama-модель для генерации (по умолчанию: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Показывать найденные чанки для каждого запроса",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Одиночный вопрос (без интерактивного режима)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("Загрузка модели эмбеддингов…")
    embedding_model = SentenceTransformerEmbeddings(EMBEDDING_MODEL)

    print("Загрузка векторного индекса…")
    vector_store = load_index(embedding_model)
    count = vector_store._collection.count()
    print(f"Индекс загружен: {count} векторов.\n")

    if args.query:
        # Одиночный запрос (удобно для тестирования)
        answer = rag_query(
            vector_store,
            args.query,
            model=args.model,
            verbose=args.verbose,
        )
        print("Ответ:\n")
        print(answer)
    else:
        repl(vector_store, model=args.model)


if __name__ == "__main__":
    main()
