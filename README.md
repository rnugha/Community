# SDU Chatbot with LangChain, LangGraph and DeepSeek

Интерактивный чат-бот, который отвечает на вопросы студентов, с использованием LangChain, LangGraph и Chroma для векторного поиска.

## Возможности

- Поддержка истории чата
- Извлечение контекста из PDF-документов
- Векторный поиск (Chroma + OpenAI embeddings)
- API-интерфейс через FastAPI
- Постоянное хранение эмбеддингов

---

### Deployment Instructions

1) Install poetry:

```bash
pip install poetry
```

2) Install libraries:

```bash
poetry install
```

3) Run the application:
```bash
poetry run uvicorn app:app --reload
```

To deploy the application, use one of the following commands:

For running in the foreground:

```bash
docker compose -p <PROJECT_NAME> up --build
```

For running in detached mode:

```bash
docker compose -p <PROJECT_NAME> up --build -d
```
Put desired name inplace of `<PROJECT_NAME>`.


### Structure

1. chatbot - основная логика (векторный поиск, история чата, генерация ответа)
2. app.py - REST API с /ask endpoint