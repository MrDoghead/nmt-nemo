#!/bin/bash

nemo_model='./model_bin2/nmt_en_zh_transformer6x6.nemo'
encoder_onnx='./model_bin2/nmt_en_zh_transformer6x6_encoder.onnx'
decoder_init_onnx='./model_bin2/nmt_en_zh_transformer6x6_decoder_init.onnx'
decoder_onnx='./model_bin2/nmt_en_zh_transformer6x6_decoder.onnx'

python utils/nemo2onnx.py \
	--pt_model=${nemo_model} \
	--encoder=${encoder_onnx} \
	--decoder_init=${decoder_init_onnx} \
	--decoder=${decoder_onnx}

