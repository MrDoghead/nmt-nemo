ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/nmt-nemo
WORKDIR /workspace/nmt-nemo
RUN pip install --no-cache-dir -r requirements.txt

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
