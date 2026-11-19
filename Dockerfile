FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sop_harness ./sop_harness
COPY apps ./apps
COPY server ./server
COPY scripts ./scripts

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/healthz')" || exit 1

# The API token is supplied at run time (-e ANTHROPIC_API_KEY=...) or pasted into the UI.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
