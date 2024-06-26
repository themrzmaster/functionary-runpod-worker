FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set timezone to UTC to avoid interactive prompts
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.9 and pip
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install PyTorch 2.1.1 with CUDA 11.8 support
RUN python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install torch==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Reinstall transformers to ensure compatibility with PyTorch and CUDA
RUN python3.9 -m pip install transformers==4.22.0

COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.9 -m pip install --upgrade -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 

ENV PYTHONPATH="/:/vllm-workspace"

# Install git
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y git

COPY src/download_model.py /download_model.py

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3.9 /download_model.py; \
    fi

# Add source files
COPY src /src
WORKDIR /src
# Remove download_model.py
RUN rm /download_model.py

# Start the handler
CMD ["python3.9", "handler.py"]
