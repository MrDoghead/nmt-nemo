import torch
import numpy as np
import nemo.collections.nlp as nemo_nlp
import onnx

def export_encoder(encoder):
    print('Encoder:',type(encoder))
    # export encoder onnx
    encoder_onnx_path = './model_bin/nmt_en_zh_transformer6x6_encoder.onnx'
    encoder.export(output=encoder_onnx_path)
    # validate model
    encoder_onnx_model = onnx.load(encoder_onnx_path)
    onnx.checker.check_model(encoder_onnx_model)
    print(f'Encoder is good and model saved at {encoder_onnx_path}')

def export_decoder_init(decoder):
    print('init decoder:',type(decoder))
    batch_size = 16
    seq_len = 128
    hidden_size = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
    decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
    encoder_states = torch.randn(size=(batch_size, seq_len, hidden_size), device=device)
    encoder_mask = torch.randint(low=1, high=2, size=(batch_size, seq_len), device=device, dtype=torch.int32)
    inputs_init = tuple([decoder_states, decoder_mask, encoder_states, encoder_mask])
    decoder_onnx_init_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx'
    decoder.forward = decoder.init_forward
    decoder.eval()
    torch.onnx.export(
            decoder,
            inputs_init,
            decoder_onnx_init_path,
            export_params=True,
            opset_version=12,
            verbose = False,
            do_constant_folding=True,
            input_names=['decoder_states','decoder_mask','encoder_states','encoder_mask'],
            output_names=['cached_mens'],
            dynamic_axes={
                'decoder_states':{0:'batch_size'},
                'decoder_mask':{0:'batch_size'},
                'encoder_states':{0:'batch_size',1:'seq_len'},
                'encoder_mask':{0:'batch_size',1:'seq_len'},
                'cached_mens':{1:'batch_size'}
                }
    )
    # validate model
    decoder_onnx_init_model = onnx.load(decoder_onnx_init_path)
    onnx.checker.check_model(decoder_onnx_init_model)
    print(f'Decoder_init is good and model saved at {decoder_onnx_init_path}')

def export_decoder_non_init(decoder):
    print('non_init decoder:',type(decoder))
    batch_size = 4*16
    seq_len = 128
    hidden_size = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder_onnx_path = './model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
    decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
    decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
    encoder_states = torch.randn(size=(batch_size, seq_len, hidden_size), device=device)
    encoder_mask = torch.randint(low=1, high=2, size=(batch_size, seq_len), device=device, dtype=torch.int32)
    decoder_mems =  torch.randn(size=(7, batch_size, 1, hidden_size), device=device)
    inputs = tuple([decoder_states, decoder_mask, encoder_states, encoder_mask, decoder_mems])
    decoder.forward = decoder.non_init_forward
    decoder.eval()
    torch.onnx.export(
            decoder,
            inputs,
            decoder_onnx_path,
            export_params=True,
            opset_version=12,
            verbose = False,
            do_constant_folding=True,
            input_names=['decoder_states','decoder_mask','encoder_states','encoder_mask','decoder_mems'],
            output_names=['cached_mens'],
            dynamic_axes={
                'decoder_states':{0:'batch_size'},
                'decoder_mask':{0:'batch_size'},
                'encoder_states':{0:'batch_size',1:'seq_len'},
                'encoder_mask':{0:'batch_size',1:'seq_len'},
                'decoder_mems':{1:'batch_size',2:'dec_len'},
                'cached_mens':{1:'batch_size',2:'dec_len'}
                }
    )

    # validate model
    decoder_onnx_model = onnx.load(decoder_onnx_path)
    onnx.checker.check_model(decoder_onnx_model)
    print(f'Decoder_non_init is good and model saved at {decoder_onnx_path}')

if __name__=='__main__':
    # load model
    nemo_path = './model_bin/nmt_en_zh_transformer6x6.nemo'
    model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=nemo_path)

    #export_encoder(encoder=model.encoder)
    #export_decoder_init(decoder=model.decoder._decoder)
    export_decoder_non_init(decoder=model.decoder._decoder)
