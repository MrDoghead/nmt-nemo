import tensorrt as trt
import numpy as np
import torch
from modules.trt_helper import EncWrapper, DecInitWrapper, DecWrapper

device = torch.device('cpu')
beam_size = 4

def mask_padded_tokens(tokens, pad_id):
    mask = np.array((tokens != pad_id),dtype=np.int32)
    #mask = tokens != pad_id
    return mask

def encode(enc_trt_path):
    enc_wrapper = EncWrapper(trt_path=enc_trt_path)

    batch_size = 2
    hidden_size = 1024
    pad_id = 0
    inputs = [[2, 11699, 6696, 12649, 4669, 758, 5879, 211, 5880, 210, 3], [2, 21004, 229, 90, 1412, 792, 219, 316, 3239, 6980, 262, 1105, 96, 147, 1355, 1401, 5880, 3]] 
    max_len = max(len(txt) for txt in inputs)
    src_ids = np.ones((len(inputs), max_len),dtype=np.int32) * pad_id
    for i, txt in enumerate(inputs):
        src_ids[i][: len(txt)] = txt
    src_mask = np.array((src_ids != pad_id),dtype=np.int32)
    shape_of_output = (batch_size, max_len, hidden_size)
    
    # inference
    print('===== encoder =====')
    print('src_ids:',src_ids)
    print('src_mask:',src_mask)
    print('shape_of_output:',shape_of_output)
    enc_outputs = enc_wrapper.do_inference(
            src_ids,
            src_mask,
            shape_of_output,
            batch_size)
    last_hidden_states = enc_outputs[0].reshape(shape_of_output)
    print('enc_outputs:',last_hidden_states)
    print('enc_outputs shape:',last_hidden_states.shape)
    print()
    encoder_hidden_states = torch.from_numpy(last_hidden_states).to(device)
    encoder_input_mask = torch.from_numpy(src_mask).to(device)
    return encoder_hidden_states, encoder_input_mask

def decode_init(dec_trt_path,encoder_states=None,encoder_mask=None):
    dec_wrapper = DecInitWrapper(trt_path=dec_trt_path)
    if encoder_states is None:
        encoder_states = torch.randn(size=(2, 16, 1024), device=device)
    if encoder_mask is None:
        encoder_mask = torch.randint(low=1, high=2, size=(2, 16, 1024), device=device, dtype=torch.int32)
    batch_size, src_length, hidden_size = encoder_states.shape
    decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
    decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
    shape_of_output = (7,batch_size,1,1024)

    #torch to numpy
    decoder_states = decoder_states.numpy()
    decoder_mask = decoder_mask.numpy()
    encoder_states = encoder_states.numpy()
    encoder_mask = encoder_mask.numpy()

    print('===== decode init =====')
    print('decoder_states:',decoder_states.shape)
    print('decoder_mask:',decoder_mask.shape)
    print('encoder_states:',encoder_states.shape)
    print('encoder_mask:',encoder_mask.shape)
    print('shape_of_output:',shape_of_output)
    dec_output = dec_wrapper.do_inference(
            decoder_states,
            decoder_mask,
            encoder_states,
            encoder_mask,
            shape_of_output,
            batch_size)
    decoder_mems = dec_output[0].reshape(shape_of_output)
    print('decoder_mems:',decoder_mems)
    print('decoder_mems shape:',decoder_mems.shape)
    print()
    decoder_mems = torch.from_numpy(decoder_mems).to(device)
    return decoder_mems

def decode(dec_trt_path, decoder_mems=None, encoder_states=None, encoder_mask=None):
    dec_wrapper = DecWrapper(trt_path=dec_trt_path)
    if encoder_states is None:
        encoder_states = torch.randn(size=(4, 16, 1024), device=device)
    if encoder_mask is None:
        encoder_mask = torch.randint(low=1, high=2, size=(4, 16, 1024), device=device, dtype=torch.int32)
    batch_size, src_length, hidden_size = encoder_states.shape
    decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
    decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
    if decoder_mems is None:
        decoder_mems =  torch.randn(size=(7, batch_size, 1, hidden_size), device=device)
    shape_of_output = (7, decoder_mems.shape[1], decoder_mems.shape[2]+1, decoder_mems.shape[3])

    # torch to numpy
    decoder_states = decoder_states.numpy()
    decoder_mask = decoder_mask.numpy().astype(np.int32)
    encoder_states = encoder_states.numpy()
    encoder_mask = encoder_mask.numpy().astype(np.int32)
    decoder_mems = decoder_mems.numpy()
    
    # inference
    print('===== decode =====')
    print('decoder_states:',decoder_states.shape)
    print('decoder_mask:',decoder_mask.shape)
    print('encoder_states:',encoder_states.shape)
    print('encoder_mask:',encoder_mask.shape)
    print('decoder_mems:',decoder_mems.shape)
    print('shape_of_output:',shape_of_output)
    dec_output = dec_wrapper.do_inference(
            decoder_states,
            decoder_mask,
            encoder_states,
            encoder_mask,
            decoder_mems,
            shape_of_output,
            batch_size)
    decoder_mems = dec_output[0].reshape(shape_of_output)
    print('decoder_mems:',decoder_mems)
    print('decoder_mems shape:',decoder_mems.shape)
    print()
    return decoder_mems

def main():
    enc_trt_path = './model_bin/nmt_en_zh_transformer6x6_encoder_fp32.trt'
    dec_init_trt_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init_fp32.trt'
    dec_trt_path = './model_bin/nmt_en_zh_transformer6x6_decoder_fp32.trt'

    encoder_states,encoder_mask = encode(enc_trt_path)
    decoder_mems = decode_init(dec_init_trt_path,encoder_states,encoder_mask)

    _,src_length,hidden_size = encoder_states.shape
    encoder_states = encoder_states.repeat(1, beam_size, 1).view(-1, src_length, hidden_size)
    encoder_mask = encoder_mask.repeat(1, beam_size).view(-1, src_length)
    #for j in range(len(decoder_mems)):
    #        decoder_mems[j] = decoder_mems[j].repeat(beam_size, 1, 1)
    decoer_mems2 = decode(dec_trt_path, decoder_mems=None, encoder_states=encoder_states, encoder_mask=encoder_mask)

if __name__ == '__main__':
    main()

