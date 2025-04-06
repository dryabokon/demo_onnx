FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y sudo && apt-get clean
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN sudo apt-get update && sudo apt-get install -y \
    mc \
    nano \
    wget \
    curl \
    git \
    unzip \
    python3-pip \
    && sudo apt-get clean
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libcublas-11-8 \
    libcudnn8 \
    libnvinfer8 \
    libnvinfer-plugin8 \
    && apt-get clean
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN apt update && apt install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    libopencv-dev \
    pkg-config
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz
RUN tar -xzf onnxruntime-linux-x64-gpu-1.21.0.tgz
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
RUN pip install --upgrade pip && pip install ultralytics onnxruntime scikit-image tabulate
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
ENV ONNXRUNTIME_DIR=/opt/onnxruntime-linux-x64-gpu-1.21.0
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$LD_LIBRARY_PATH
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
CMD ["bash"]


