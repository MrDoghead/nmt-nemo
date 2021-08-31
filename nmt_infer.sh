#!/bin/sh

model_path='./model_bin/nmt_en_zh_transformer6x6.nemo'
src_path='./data/test_1k.en'
tgt_path='./data/pred_nemo.zh'
src_lang='en'
tgt_lang='zh'
batch_size=1
beam_size=4

# load model
if [ ! -f "$model_path" ]; then
    echo "download model from ngc..."
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_zh_transformer6x6/versions/1.0.0rc1/zip -O nmt_en_zh_transformer6x6_1.0.0rc1.zip
    unzip nmt_en_zh_transformer6x6_1.0.0rc1.zip
    rm nmt_en_zh_transformer6x6_1.0.0rc1.zip
    echo "model saved at ${model_path}"
fi

# check output path
if [ ! -f "$tgt_path" ]; then
    touch "$tgt_path"
fi

# main
python nmt_transformer_infer.py \
    --model=${model_path} \
    --srctext=${src_path} \
    --tgtout=${tgt_path} \
    --batch_size=${batch_size} \
    --beam_size=${beam_size} \
    --target_lang ${tgt_lang} \
    --source_lang ${src_lang}
