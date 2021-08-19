import onnx
import torch
import onnxruntime

#model_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx' 
model_path = './model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
session = onnxruntime.InferenceSession(model_path)
session.get_modelmeta()
input_names = [each.name for each in session.get_inputs()]
output_names = [each.name for each in session.get_outputs()]
print('input names:',input_names)
print('output names:',output_names)

"""
batch_size = 2
seq_len = 18
hidden_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
decoder_states = torch.randn(size=(batch_size, 1, hidden_size), device=device)
decoder_mask = torch.randint(low=1, high=2, size=(batch_size, 1), device=device, dtype=torch.int32)
encoder_states = torch.randn(size=(batch_size, seq_len, hidden_size), device=device)
encoder_mask = torch.randint(low=1, high=2, size=(batch_size, seq_len), device=device, dtype=torch.int32)

"""

batch_size = 2
seq_len = 18
hidden_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
decoder_states = torch.randn(size=(4*batch_size, 1, hidden_size), device=device)
decoder_mask = torch.randint(low=1, high=2, size=(4*batch_size, 1), device=device, dtype=torch.int32)
encoder_states = torch.randn(size=(4*batch_size, seq_len, hidden_size), device=device)
encoder_mask = torch.randint(low=1, high=2, size=(4*batch_size, seq_len), device=device, dtype=torch.int32)
decoder_mems =  torch.randn(size=(7, 4*batch_size, 1, hidden_size), device=device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

print('decoder_states',decoder_states.shape)
print('decoder_mask',decoder_mask.shape)
print('encoder_states',encoder_states.shape)
print('encoder_mask',encoder_mask.shape)
print('decoder_mems',decoder_mems.shape)
res = session.run(None,
        {
            'decoder_states':to_numpy(decoder_states), 
            'decoder_mask':to_numpy(decoder_mask), 
            'encoder_states':to_numpy(encoder_states),
            'encoder_mask':to_numpy(encoder_mask),
            'decoder_mems':to_numpy(decoder_mems)
        }
)
print('res:', res)
print('len of res', len(res))
print('res[0] shape:', res[0].shape)
