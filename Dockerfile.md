# Imagem base enxuta e compatível com linux/amd64
FROM python:3.12-slim-bookworm

# Define diretório de trabalho
WORKDIR /app

# Instala Poetry
RUN pip install --no-cache-dir poetry

# Copia apenas arquivos de dependência primeiro (para cache eficiente)
COPY app/pyproject.toml app/poetry.lock* ./

# Instala dependências no sistema (sem criar venv dentro do container)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copia o restante da aplicação (conteúdo da pasta app direto para /app)
COPY app/ .

# Exposição da porta padrão do Uvicorn
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
