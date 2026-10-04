# ModerationKit — single-container deploy.
# Builds an image that runs the FastAPI moderation service on port 8000.
# To run the Gradio UI instead, override CMD: `docker run ... app.py`

FROM python:3.11-slim

# System deps for OpenCV (libgl, libglib) and HTTPS downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY *.py ./
COPY integrations/ ./integrations/

# Pre-download both models at build time so cold starts are fast.
# - NudeNet 640m via our download script
# - Falconsai via transformers cache warm-up
RUN python download_models.py
RUN python -c "from nsfw_classifier import NSFWClassifier; NSFWClassifier()"
# The models above are the ones that run: no updates from the internet at startup.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Default: run the FastAPI moderation service. Override CMD to run the UI.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
