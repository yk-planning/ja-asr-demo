FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-train.txt ./
ARG WITH_TRAIN=0
RUN pip install -r requirements.txt \
 && if [ "$WITH_TRAIN" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-train.txt; \
    fi

COPY server.py live.py train.py glossary.txt ./
COPY static ./static
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
