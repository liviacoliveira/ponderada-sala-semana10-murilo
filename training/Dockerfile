FROM python:3.11-slim
WORKDIR /app
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ ./common/
COPY training/train.py ./train.py
ENV DATA_PATH=/data/btc_usd_daily.csv MODEL_PATH=/models/model.joblib
CMD ["python", "train.py"]
