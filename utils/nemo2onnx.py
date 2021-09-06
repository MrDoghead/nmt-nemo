import torch
import numpy as np
import nemo.collections.nlp as nemo_nlp
import onnx
import argparse
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f'ONNX is good and saved at {onnx_path}')

def export_from_nemo(nemo_model, onnx_path):
    print('Nemo model:',type(nemo_model))
    nemo_model.export(output=onnx_path)
    # validate model
    validate(onnx_path)

def export_from_pt(pt_model, onnx_path, dummy_inputs, output_names, dynamic_axes):
    print('Pytorch model:',type(pt_model))
    pt_model.eval()
    torch.onnx.export(
            pt_model,
            tuple(dummy_inputs.values()),
            onnx_path,
            export_params = True,
            opset_version = 12,
            verbose = False,
            do_constant_folding = True,
            input_names = list(dummy_inputs.keys()),
            output_names = output_names,
            dynamic_axes= dynamic_axes
    )
    # validate model
    validate(onnx_path)

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--pt_model', required=True,
                        help='pt model path')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the save Encoder ONNX')
    parser.add_argument('--decoder_init', type=str, default="",
                        help='full path to the save Decoder Init ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the save Decoder ONNX')

    return parser

def main():
    parser = argparse.ArgumentParser(
        description='Export from nemo to ONNX for nmt-nemo')
    parser = parse_args(parser)
    args = parser.parse_args()

    print(f'Loading nemo model from: {args.pt_model}')
    model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=args.pt_model)

    # dummy config
    batch_size = 16
    seq_len = 128
    hidden_size = 1024

    if args.encoder != "":
        print('Exporting encoder...')
        export_from_nemo(
                nemo_model=model.encoder, 
                onnx_path=args.encoder
                )

    if args.decoder_init != "":
        print('Exporting decoder_init...')
        decoder_init = model.decoder._decoder
        decoder_init.forward = decoder_init.init_forward
        decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
        decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
        encoder_states = torch.randn(size=(batch_size, seq_len, hidden_size), device=device)
        encoder_mask = torch.randint(low=1, high=2, size=(batch_size, seq_len), device=device, dtype=torch.int32)
        dummy_inputs = {
                "decoder_states": decoder_states,
                "decoder_mask": decoder_mask,
                "encoder_states": encoder_states,
                "encoder_mask": encoder_mask,
                }
        output_names = ['cached_mems']
        dynamic_axes={
                'decoder_states':{0:'batch_size'},
                'decoder_mask':{0:'batch_size'},
                'encoder_states':{0:'batch_size',1:'seq_len'},
                'encoder_mask':{0:'batch_size',1:'seq_len'},
                'cached_mems':{1:'batch_size'},
                }
        print("===== ONNX INFO =====")
        print("dummy_inputs:",dummy_inputs)
        print("output_names:",output_names)
        print("dynamic_axes:",dynamic_axes)
        export_from_pt(
                pt_model=decoder_init, 
                onnx_path=args.decoder_init, 
                dummy_inputs=dummy_inputs, 
                output_names=output_names, 
                dynamic_axes=dynamic_axes,
                )

    if args.decoder != "":
        print("Exporting decoder...")
        decoder = decoder=model.decoder._decoder
        decoder.forward = decoder.non_init_forward
        decoder_states = torch.randn(size=(batch_size*4, 1, hidden_size), device=device)
        decoder_mask = torch.randint(low=1, high=2, size=(batch_size*4, 1), device=device, dtype=torch.int32)
        encoder_states = torch.randn(size=(batch_size*4, seq_len, hidden_size), device=device)
        encoder_mask = torch.randint(low=1, high=2, size=(batch_size*4, seq_len), device=device, dtype=torch.int32)
        decoder_mems =  torch.randn(size=(7, batch_size*4, 1, hidden_size), device=device)
        dummy_inputs = {
                "decoder_states": decoder_states,
                "decoder_mask": decoder_mask,
                "encoder_states": encoder_states,
                "encoder_mask": encoder_mask,
                "decoder_mems": decoder_mems,
                }
        output_names = ['cached_mems']
        dynamic_axes={
                'decoder_states':{0:'batch_size'},
                'decoder_mask':{0:'batch_size'},
                'encoder_states':{0:'batch_size',1:'seq_len'},
                'encoder_mask':{0:'batch_size',1:'seq_len'},
                'decoder_mems':{1:'batch_size',2:'dec_len'},
                'cached_mems':{1:'batch_size',2:'dec_len'}
                }
        print("===== ONNX INFO =====")
        print("dummy_inputs:",dummy_inputs)
        print("output_names:",output_names)
        print("dynamic_axes:",dynamic_axes)
        export_from_pt(
                pt_model=decoder,
                onnx_path=args.decoder,
                dummy_inputs=dummy_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                )

if __name__=='__main__':
    main()
