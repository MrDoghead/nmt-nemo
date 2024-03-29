#!/bin/sh

set -x

enc_onnx_path='./model_bin/nmt_en_zh_transformer6x6_encoder.onnx'
enc_trt_path='./model_bin/nmt_en_zh_transformer6x6_encoder_fp16.engine'
enc_min_shape="'input_ids':1x1,'encoder_mask':1x1"
enc_opt_shape="'input_ids':8x128,'encoder_mask':8x128"
enc_max_shape="'input_ids':16x256,'encoder_mask':16x256"
enc_inf_shape=${enc_min_shape}

dec_onnx_init_path='./model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx'
dec_trt_init_path='./debug/nmt_en_zh_transformer6x6_decoder_init_fp32.engine'
dec_min_init_shape="'decoder_states':1x1x1024,'decoder_mask':1x1,'encoder_states':1x1x1024,'encoder_mask':1x1"
dec_opt_init_shape="'decoder_states':8x1x1024,'decoder_mask':8x1,'encoder_states':8x128x1024,'encoder_mask':8x128"
dec_max_init_shape="'decoder_states':16x1x1024,'decoder_mask':16x1,'encoder_states':16x256x1024,'encoder_mask':16x256"
dec_inf_init_shape=${dec_min_init_shape}

dec_onnx_path='./model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
dec_trt_path='./model_bin/nmt_en_zh_transformer6x6_decoder_fp16.engine'
dec_min_shape="'decoder_states':1x1x1024,'decoder_mask':1x1,'encoder_states':1x1x1024,'encoder_mask':1x1,'decoder_mems':7x1x1x1024"
dec_opt_shape="'decoder_states':32x1x1024,'decoder_mask':32x1,'encoder_states':32x128x1024,'encoder_mask':32x128,'decoder_mems':7x32x128x1024"
dec_max_shape="'decoder_states':64x1x1024,'decoder_mask':64x1,'encoder_states':64x256x1024,'encoder_mask':64x256,'decoder_mems':7x64x256x1024"
dec_inf_shape=${dec_min_shape}

#trtexec --onnx=${enc_onnx_path} \
#    --explicitBatch \
#    --minShapes=${enc_min_shape} \
#    --optShapes=${enc_opt_shape} \
#    --maxShapes=${enc_max_shape} \
#    --fp16 \
#    --shapes=${enc_inf_shape} \
#    --saveEngine=${enc_trt_path}

#trtexec --onnx=${dec_onnx_init_path} \
#    --explicitBatch \
#    --minShapes=${dec_min_init_shape} \
#    --optShapes=${dec_opt_init_shape} \
#    --maxShapes=${dec_max_init_shape} \
#    --shapes=${dec_inf_init_shape} \
#    --saveEngine=${dec_trt_init_path}

#trtexec --onnx=${dec_onnx_path} \
#    --explicitBatch \
#    --minShapes=${dec_min_shape} \
#    --optShapes=${dec_opt_shape} \
#    --maxShapes=${dec_max_shape} \
#    --shapes=${dec_inf_shape} \
#    --saveEngine=${dec_trt_path} \
#    --workspace=8192

### do inference ###

trtexec --loadEngine=${dec_trt_init_path} \
        --shapes=${dec_inf_init_shape} \
        --iterations=10 \
	--dumpProfile
