FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY web_app.py /app/web_app.py
COPY dblp_match.py /app/dblp_match.py
COPY runtime_store.py /app/runtime_store.py
COPY templates /app/templates
COPY static /app/static

VOLUME ["/data"]
EXPOSE 8092

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8092"]
