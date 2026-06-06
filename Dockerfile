# Imagem base enxuta e compatível com linux/amd64
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY app/pyproject.toml app/poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
