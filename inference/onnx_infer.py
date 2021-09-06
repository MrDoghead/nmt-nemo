import onnx
import torch
import onnxruntime

# enc_model_path = './model_bin/nmt_en_zh_transformer6x6_encoder.onnx'
# enc_session = onnxruntime.InferenceSession(enc_model_path)
# enc_session.get_modelmeta()
# print('enc input names:',[each.name for each in enc_session.get_inputs()])
# print('enc output names:',[each.name for each in enc_session.get_outputs()])

# dec_init_model_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx' 
dec_init_model_path = './debug/nmt_en_zh_transformer6x6_decoder_init.onnx'
dec_init_session = onnxruntime.InferenceSession(dec_init_model_path)
dec_init_session.get_modelmeta()
print('dec_init input names:',[each.name for each in dec_init_session.get_inputs()])
print('dec_init output names:',[each.name for each in dec_init_session.get_outputs()])

# dec_model_path = './model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
# dec_session = onnxruntime.InferenceSession(dec_model_path)
# dec_session.get_modelmeta()
# print('dec input names:',[each.name for each in dec_session.get_inputs()])
# print('dec output names:',[each.name for each in dec_session.get_outputs()])

"""
batch_size = 2
seq_len = 18
hidden_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
encoder_states = torch.randn(size=(batch_size, seq_len, hidden_size), device=device)
encoder_mask = torch.randint(low=1, high=2, size=(batch_size, seq_len), device=device, dtype=torch.int32)

decoder_states2 = torch.randn(size=(4*batch_size, 1, hidden_size), device=device)
decoder_mask2 = torch.randint(low=1, high=2, size=(4*batch_size, 1), device=device, dtype=torch.int32)
encoder_states2 = torch.randn(size=(4*batch_size, seq_len, hidden_size), device=device)
encoder_mask2 = torch.randint(low=1, high=2, size=(4*batch_size, seq_len), device=device, dtype=torch.int32)
decoder_mems =  torch.randn(size=(7, 4*batch_size, 1, hidden_size), device=device)
"""

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# enc_input = torch.load('./debug/enc_input.pt')
# input_ids = enc_input['input_ids']
# encoder_mask = enc_input['encoder_mask'].to(torch.int64)

# enc_res = enc_session.run(None,
        # {
            # 'input_ids':to_numpy(input_ids),
            # 'encoder_mask':to_numpy(encoder_mask),
        # }
# )
# encoder_states = enc_res[0]

data = torch.load('./debug/dec_init_input.pt',map_location="cuda:0" if torch.cuda.is_available() else "cpu")
decoder_states = data['decoder_states']
decoder_mask = data['decoder_mask'].to(torch.int32)
encoder_states = data['encoder_states']
encoder_mask = data['encoder_mask'].to(torch.int32)
# encoder_mask = encoder_mask.to(torch.int32)

# print('encoder_states:',encoder_states)
dec_res = dec_init_session.run(None,
        {
            'decoder_states':to_numpy(decoder_states), 
            'decoder_mask':to_numpy(decoder_mask), 
            'encoder_states':to_numpy(encoder_states),
            'encoder_mask':to_numpy(encoder_mask),
        }
)
cached_mems = dec_res[0]
print('dec_res:', cached_mems)
print('cached_mems shape:', cached_mems.shape)
