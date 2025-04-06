#!/bin/bash

cd /home/runtime_cpp

ln -sf /opt/onnxruntime-linux-x64-gpu-1.21.0 onnxruntime-linux-x64-gpu-1.15.1

rm -rf build
mkdir -p build
cd build

cmake .. \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.21.0 \
  -DLINUX=TRUE \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
  -DCUDA_INCLUDE_DIRS=/usr/local/cuda-11.8/include \
  -DCUDA_CUDART_LIBRARY=/usr/local/cuda-11.8/lib64/libcudart.so \
  -DCMAKE_CXX_FLAGS="-I/opt/onnxruntime-linux-x64-gpu-1.21.0/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/onnxruntime-linux-x64-gpu-1.21.0/lib -lonnxruntime"

make

# run it!
cd ..
cp ./build/Yolov8OnnxRuntimeCPPInference .
./Yolov8OnnxRuntimeCPPInference


