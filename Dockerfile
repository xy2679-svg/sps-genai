FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir uv

RUN uv sync --frozen

RUN uv run python -m spacy download en_core_web_lg

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]