FROM python:3.11-slim

WORKDIR /app

# system dependencies (important for uv)
RUN apt-get update && apt-get install -y curl

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# make uv available in PATH
ENV PATH="/root/.local/bin:$PATH"

# copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# install dependencies using uv
RUN uv sync --frozen

# copy app source
COPY . .

# expose FastAPI port
EXPOSE 8000

# run server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
