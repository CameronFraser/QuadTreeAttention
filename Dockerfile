FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive TZ="Etc/UTC" apt-get install -y tzdata

RUN apt-get install -y \
    build-essential \
    git \
    wget \
    python3 \
    python3-pip \
    python-is-python3 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1

RUN mkdir -p /workspace

WORKDIR /workspace

COPY ./FeatureMatching ./FeatureMatching
COPY ./QuadTreeAttention ./QuadTreeAttention
COPY ./Makefile ./Makefile

RUN pip install --upgrade pip

WORKDIR /workspace/QuadTreeAttention

RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install kornia_moons
RUN pip install opencv-contrib-python

RUN python setup.py install

WORKDIR /workspace/FeatureMatching
RUN python setup.py install

WORKDIR /workspace

RUN make download-weights