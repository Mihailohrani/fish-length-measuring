FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PYTHONPATH="/app/src"

EXPOSE 2718

CMD ["uv", "run", "marimo", "run", "src/fish_project/notebook.py", "--headless", "--host", "0.0.0.0", "--port", "2718"]
