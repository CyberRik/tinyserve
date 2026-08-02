FROM python:3.11-slim

# llama-cpp-python has no prebuilt wheel for most platforms and compiles
# its C++ core via cmake at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY tinyserve ./tinyserve
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "tinyserve.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
