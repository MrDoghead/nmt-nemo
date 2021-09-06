#!/bin/bash

output="./model_bin"
encoder="./model_bin/nmt_en_zh_transformer6x6_encoder.onnx"
decoder_init="./model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx"
decoder="./model_bin/nmt_en_zh_transformer6x6_decoder.onnx"
maxBatchSize=16
calib=""

python utils/onnx2trt.py \
    --output=${output} \
    --encoder=${encoder} \
    --decoder_init=${decoder_init} \
    --decoder=${decoder} \
    --max_batch_size=${maxBatchSize} \
