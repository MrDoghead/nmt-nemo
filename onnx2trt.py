import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # This logger is required to build an engine

def GiB(val):
    return val * 1 << 30

def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, fp16_mode=False, int8_mode=False, mode=0, calib=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    """
    mode: choose dynamic mode
        0: encoder; 1: emb+dec; 2:deocder_init; 3: decoder_non_init; else static
    """
    # onnx not support implicit batch and must specify the explicit batch
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = GiB(8)
        builder.max_batch_size = max_batch_size
        if fp16_mode:
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True
            config.set_flag(trt.BuilderFlag.FP16)
        if int8_mode:
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
            raise NotImplementedError

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        # dynamic inputs setting
        profile = builder.create_optimization_profile()
        if mode==0:
            profile.set_shape("input_ids",          (1,1),          (8,128),           (16,256))
            profile.set_shape("encoder_mask",       (1,1),          (8,128),           (16,256))
        elif mode==1:
            profile.set_shape("input_ids",          (1,1),          (8,128),           (16,256))
            profile.set_shape("decoder_mask",       (1,1),          (8,128),           (16,256))
            profile.set_shape("encoder_mask",       (1,1,1024),     (8,128,1024),      (16,256,1024))
            profile.set_shape("encoder_embeddings", (1,1),          (8,128),           (16,256))
        elif mode==2:
            profile.set_shape("decoder_states",     (1,1,1024),     (8,1,1024),        (16,1,1024))
            profile.set_shape("decoder_mask",       (1,1),          (8,1),             (16,1))
            profile.set_shape("encoder_states",     (1,1,1024),     (8,128,1024),      (16,256,1024))
            profile.set_shape("encoder_mask",       (1,1),          (8,128),           (16,256))
        elif mode==3:
            profile.set_shape("decoder_states",     (1,1,1024),     (4*8,1,1024),        (4*16,1,1024))
            profile.set_shape("decoder_mask",       (1,1),          (4*8,1),             (4*16,1))
            profile.set_shape("encoder_states",     (1,1,1024),     (4*8,128,1024),      (4*16,256,1024))
            profile.set_shape("encoder_mask",       (1,1),          (4*8,128),           (4*16,256))
            profile.set_shape("decoder_mems",       (7,1,1,1024),   (7,4*8,128,1024),    (7,4*16,256,1024))
        else:
            print('Using static inputs!')
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network,config)
        if engine is None:
            print("Failed to create the engine")
            return None
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine saved at {}'.format(engine_file_path))
        
        return engine

def build_encoder():
    enc_onnx_path = './model_bin/nmt_en_zh_transformer6x6_encoder.onnx'
    #enc_engine_path = './model_bin/nmt_en_zh_transformer6x6_encoder_fp32.trt'
    enc_engine_path = './model_bin/nmt_en_zh_transformer6x6_encoder_fp16.trt'
    enc_engine = build_engine(enc_onnx_path, enc_engine_path, max_batch_size=16, fp16_mode=True, int8_mode=False, mode=0)

def build_decoder_init():
    dec_onnx_init_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx'
    #dec_engine_init_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init_fp32.trt'
    dec_engine_init_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init_fp16.trt'
    dec_engine = build_engine(dec_onnx_init_path, dec_engine_init_path, max_batch_size=16, fp16_mode=True, int8_mode=False, mode=2)

def build_decoder_non_init():
    dec_onnx_path = './model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
    #dec_engine_path = './model_bin/nmt_en_zh_transformer6x6_decoder_fp32.trt'
    dec_engine_path = './model_bin/nmt_en_zh_transformer6x6_decoder_fp16.trt'
    dec_engine = build_engine(dec_onnx_path, dec_engine_path, max_batch_size=64, fp16_mode=True, int8_mode=False, mode=3)

if __name__ == '__main__':
    #build_encoder()
    #build_decoder_init()
    build_decoder_non_init()
