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

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-deps rfdetr

RUN pip3 install cython pycocotools scipy timm tqdm accelerate transformers \
    peft ninja einops pandas pylabel onnx onnxsim onnx_graphsurgeon \
    polygraphy open_clip_torch rf100vl pydantic supervision matplotlib \
    Pillow requests huggingface-hub safetensors packaging pyyaml

WORKDIR /app

RUN mkdir -p /app/models

RUN python3.9 -c "
from rfdetr import RFDETRBase
import numpy as np

print('Downloading RF-DETR Base model...')
model = RFDETRBase()

# Trigger a dummy prediction to ensure all weights are loaded
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    _ = model.predict(dummy_image, threshold=0.5)
    print('Model weights downloaded and verified successfully')
except Exception as e:
    print(f'Dummy prediction failed but model should be cached: {e}')

print('RF-DETR model setup completed')
"

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.9", "main.py"]