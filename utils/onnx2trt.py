import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import argparse
import sys

# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # This logger is required to build an engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def mark_outputs(network):
    print('network.num_layers:',network.num_layers)
    for i in range(network.num_layers-1):
        layer = network.get_layer(i)
        print(i,layer.name,layer.type)
        continue
        for j in range(layer.num_outputs):
            network.mark_output(layer.get_output(j))

def GiB(val):
    return val * 1 << 30

def build_engine(model_file, shapes, max_batch_size=16, fp16_mode=False, int8_mode=False, calib=""):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
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

        print('Loading ONNX file from path {}...'.format(model_file))
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        print('Complete parsing of ONNX file')

        # to compromise the trt fp16 optimization bugs
        #
        # mark_outputs(network)
        # sys.exit()
        #
        # idx = [400, 680, 960, 1240, 1520] # dec_init
        # idx = [210, 493, 776, 1059, 1342] # dec
        # for i in idx:
            # layer = network.get_layer(i)
            # print('layer:',i,layer.name,layer.type,layer.precision)
            # for j in range(layer.num_outputs):
                # output = layer.get_output(j)
                # print('output:',output.name,output.shape)
                # network.mark_output(output)

        # dynamic inputs setting
        profile = builder.create_optimization_profile()
        for s in shapes:
            profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network,config)
        
        return engine

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save trt engines')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder_init', type=str, default="",
                        help='full path to the Decoder Init ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the Decoder ONNX')
    parser.add_argument('--max_batch_size', type=int, default=16,
                        help='max batch size for trt inference')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')
    parser.add_argument('--int8', action='store_true',
                        help='inference with INT8')
    parser.add_argument('--calib', type=str, default="",
                        help="Calibration File for int8")

    return parser

def main():
    parser = argparse.ArgumentParser(
        description='Export from ONNX to TensorRT for nmt-nemo')
    parser = parse_args(parser)
    args = parser.parse_args()

    engine_prec = "_fp16" if args.fp16 else "_fp32"
    if args.int8:
        engine_prec = "_int8"

    # Encoder
    shapes=[{"name": "input_ids",           "min": (1,1),    "opt": (8,128),     "max": (16,256)},
            {"name": "encoder_mask",        "min": (1,1),    "opt": (8,128),     "max": (16,256)}]
    if args.encoder != "":
        print("Building Encoder ...")
        encoder_engine = build_engine(
                model_file=args.encoder, 
                shapes=shapes, 
                max_batch_size=args.max_batch_size, 
                fp16_mode=args.fp16, 
                int8_mode=args.int8,
                calib=args.calib)
        if encoder_engine is not None:
            engine_name = args.encoder.split('/')[-1].split('.')[0] + engine_prec + '.engine'
            engine_path = args.output + "/" + engine_name
            with open(engine_path, 'wb') as f:
                f.write(encoder_engine.serialize())
                print("Engine saved at",engine_path)
        else:
            print("Failed to build engine from", args.encoder)
            sys.exit()

    # DecoderInit
    shapes=[{"name": "decoder_states",      "min": (1,1,1024),  "opt": (8,1,1024),      "max": (16,1,1024)},
            {"name": "decoder_mask",        "min": (1,1),       "opt": (8,1),           "max": (16,1)},
            {"name": "encoder_states",      "min": (1,1,1024),  "opt": (8,128,1024),    "max": (16,256,1024)},
            {"name": "encoder_mask",        "min": (1,1),       "opt": (8,128),         "max": (16,256)}]
    if args.decoder_init != "":
        print("Building DecoderInit ...")
        decoder_init_engine = build_engine(
                model_file=args.decoder_init, 
                shapes=shapes, 
                max_batch_size=args.max_batch_size,
                fp16_mode=args.fp16, 
                int8_mode=args.int8,
                calib=args.calib)
        if decoder_init_engine is not None:
            engine_name = args.decoder_init.split('/')[-1].split('.')[0] + engine_prec + '.engine'
            engine_path = args.output + "/" + engine_name
            with open(engine_path, 'wb') as f:
                f.write(decoder_init_engine.serialize())
                print("Engine saved at",engine_path)
        else:
            print("Failed to build engine from", args.decoder_init)
            sys.exit()

    # Decoder
    shapes=[{"name": "decoder_states",      "min": (1,1,1024),      "opt": (4*8,1,1024),        "max": (4*16,1,1024)},
            {"name": "decoder_mask",        "min": (1,1),           "opt": (4*8,1),             "max": (4*16,1)},
            {"name": "encoder_states",      "min": (1,1,1024),      "opt": (4*8,128,1024),      "max": (4*16,256,1024)},
            {"name": "encoder_mask",        "min": (1,1),           "opt": (4*8,128),           "max": (4*16,256)},
            {"name": "decoder_mems",        "min": (7,1,1,1024),    "opt": (7,4*8,128,1024),    "max": (7,4*16,256,1024)}]
    if args.decoder != "":
        print("Building DecoderInit ...")
        decoder_engine = build_engine(
                model_file=args.decoder,
                shapes=shapes, 
                max_batch_size=args.max_batch_size*4, 
                fp16_mode=args.fp16, 
                int8_mode=args.int8,
                calib=args.calib)
        if decoder_engine is not None:
            engine_name = args.decoder.split('/')[-1].split('.')[0] + engine_prec + '.engine'
            engine_path = args.output + "/" + engine_name
            with open(engine_path, 'wb') as f:
                f.write(decoder_engine.serialize())
                print("Engine saved at",engine_path)
        else:
            print("Failed to build engine from", args.decoder)
            sys.exit()

if __name__ == '__main__':
    main()
