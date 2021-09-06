#!/bin/bash

model_path="./model_bin2"

# check output path
if [ ! -f "$model_path" ]; then
    mkdir "$model_path"
fi

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_zh_transformer6x6/versions/1.0.0rc1/zip -O nmt_en_zh_transformer6x6_1.0.0rc1.zip

unzip -d ${model_path} nmt_en_zh_transformer6x6_1.0.0rc1.zip

rm nmt_en_zh_transformer6x6_1.0.0rc1.zip
