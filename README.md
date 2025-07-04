# SDU Chatbot with LangChain, LangGraph and Chroma

Интерактивный чат-бот, который отвечает на вопросы студентов, с использованием LangChain, LangGraph и Chroma для векторного поиска.

## Возможности

- Поддержка истории чата
- Извлечение контекста из PDF-документов
- Векторный поиск (Chroma + OpenAI embeddings)
- API-интерфейс через FastAPI
- Постоянное хранение эмбеддингов

---

### Git
To clone project with submodule

```bash
git clone --recurse-submodules https://github.com/rnugha/Community.git
```

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
poetry run uvicorn app:app --reload --port 8001
```

To deploy the application, use one of the following commands:

For running in the foreground:

```bash
docker compose -p <PROJECT_NAME> up --build
```


### Structure

1. chatbot - основная логика (векторный поиск, история чата, генерация ответа)
2. app.py - REST API с /ask endpoint