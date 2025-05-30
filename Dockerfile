FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    wget \
    git \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

RUN python3.9 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app

RUN mkdir -p /app/models

RUN python3.9 -c "from rfdetr import RFDETRBase; import torch; model = RFDETRBase(); torch.save({'model_state_dict': model.model.state_dict()}, '/app/models/rfdetr_backup.pt'); print('RF-DETR model downloaded and cached')"

RUN ls -la /app/models/ && echo "Model caching completed"

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.9", "main.py"]