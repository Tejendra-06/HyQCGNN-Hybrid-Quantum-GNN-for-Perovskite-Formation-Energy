# Web app image (also the Hugging Face Space). Build from the staged folder:
#   python deploy/stage_hf_space.py
#   docker build -t hyqcgnn build/hf-space
#   docker run -p 7860:7860 hyqcgnn          ->  http://localhost:7860
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY --chown=user . .
USER user

# trains and caches the XGBoost model now, so the server starts fast (and the build fails if anything is broken)
RUN python predictor.py SrTiO3

EXPOSE 7860
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
