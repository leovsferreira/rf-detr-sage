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

RUN pip3 install opencv-python==4.8.0.74 numpy
RUN pip3 install pywaggle[all]==0.56.0
RUN pip3 pip install rfdetr
RUN pip3 install boto3 botocore

WORKDIR /app

RUN mkdir -p /app/models

RUN echo 'from rfdetr import RFDETRBase; import numpy as np; print("Downloading RF-DETR model..."); model = RFDETRBase(); dummy_image = np.zeros((480, 640, 3), dtype=np.uint8); model.predict(dummy_image, threshold=0.5); print("Model download completed")' > /tmp/download_model.py

RUN python3.9 /tmp/download_model.py && rm /tmp/download_model.py

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.9", "main.py"]