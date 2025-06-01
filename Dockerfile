FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    wget \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN python --version && pip --version && cmake --version

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-aarch64.tar.gz && \
    tar -zxvf cmake-3.27.9-linux-aarch64.tar.gz && \
    mv cmake-3.27.9-linux-aarch64 /opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/bin/cmake && \
    cmake --version

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
RUN mkdir -p /app/models

RUN python -c "from rfdetr import RFDETRBase; import torch; model = RFDETRBase(); torch.save(model, '/app/models/rfdetr_backup.pt'); print('RF-DETR model downloaded and cached')"

COPY . .

ENTRYPOINT ["python", "main.py"]