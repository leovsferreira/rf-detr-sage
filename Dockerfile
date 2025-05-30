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

RUN python3.9 -c "
from rfdetr import RFDETRBase
import torch
import os
import shutil

print('Downloading RF-DETR model...')
model = RFDETRBase()

# The model weights are typically cached in the user's home directory
# Let's find where they are and copy them to our models directory
cache_dirs = [
    os.path.expanduser('~/.cache/torch/hub/checkpoints'),
    os.path.expanduser('~/.cache/huggingface'),
    '/root/.cache/torch/hub/checkpoints',
    '/root/.cache/huggingface'
]

model_found = False
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        print(f'Checking cache directory: {cache_dir}')
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if 'rfdetr' in file.lower() or 'detr' in file.lower():
                    src = os.path.join(root, file)
                    dst = f'/app/models/{file}'
                    shutil.copy2(src, dst)
                    print(f'Copied model file: {src} -> {dst}')
                    model_found = True

# Also save the model state dict as backup
try:
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': getattr(model, 'config', None)
    }, '/app/models/rfdetr_backup.pt')
    print('Model state dict saved to /app/models/rfdetr_backup.pt')
    model_found = True
except Exception as e:
    print(f'Failed to save model state dict: {e}')

if model_found:
    print('RF-DETR model cached successfully')
else:
    print('Warning: Model files not found in expected locations, but model should still work')
"

RUN ls -la /app/models/ && echo "Model caching completed"

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.9", "main.py"]