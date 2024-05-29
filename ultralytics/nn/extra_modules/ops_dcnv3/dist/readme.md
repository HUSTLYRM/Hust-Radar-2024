python: 3.8
torch                         1.13.1+cu117
torchvision                   0.14.1+cu117

cuda:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0

nvidia-smi:
NVIDIA-SMI 525.116.04   Driver Version: 525.116.04   CUDA Version: 12.0

GPU:
RTX3090

other:
https://github.com/OpenGVLab/InternImage/releases/tag/whl_files

commad:
install: python setup.py build install
clean: python setup.py clean --all