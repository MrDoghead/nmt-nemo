import torch
import numpy as np
import nemo.collections.nlp as nemo_nlp
import onnx

# load model
nemo_path = './model_bin/nmt_en_zh_transformer6x6.nemo'
model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=nemo_path)
decoder = model.decoder._decoder
decoder.eval()
print('decoder:',decoder)

#decoder_onnx_path = './model_bin/nmt_en_zh_transformer6x6_decoder_init.onnx'
decoder_onnx_path = './model_bin/nmt_en_zh_transformer6x6_decoder.onnx'
decoder.export(output=decoder_onnx_path)
print('exported to',decoder_onnx_path)
