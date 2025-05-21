FROM python:3.9-slim

# Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Установка зависимостей
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Копируем весь проект
COPY . .

# Переменные окружения по умолчанию
ENV OPENAI_API_KEY=""
ENV CHROMA_PATH=/app/chroma_store
ENV DATA_PATH=/app/data

# Создаём директории, если нет
RUN mkdir -p /app/data /app/chroma_store

# Запускаем FastAPI (можно поменять на ui.py при желании)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
