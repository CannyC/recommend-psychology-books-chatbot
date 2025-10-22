FROM python:3.13-slim
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY public ./public
COPY chainlit.md ./
COPY db ./db

COPY main.py tool_book_recommender_async.py ./

CMD uv run chainlit run -h --host 0.0.0.0 main.py
