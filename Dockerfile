FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY dblp_builder /app/dblp_builder
COPY pc-members.csv /app/pc-members.csv
COPY templates /app/templates
COPY static /app/static

VOLUME ["/data"]
EXPOSE 8091

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091"]
