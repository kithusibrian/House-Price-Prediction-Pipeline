# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Render forwards its configured web-service port to the container.
EXPOSE 8000

# Set HF_TOKEN in Render; do not bake credentials into the image.
ENV HF_REPO_ID=briankm2004/housing-regression-serving-artifacts

# Command to run API with Uvicorn
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]