FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*
RUN pip install "numpy<2" \
    torch==2.2.0 torch_xla==2.2.0 torchvision==0.17.0 \
    -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/tpu_v6e/

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python3", "train.py"]
