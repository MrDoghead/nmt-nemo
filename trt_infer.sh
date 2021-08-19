#!/bin/sh

src_file='./data/test.en'
tgt_file='./data/test.zh'
tgt_save='./data/pred.zh'
enc_engine='./model_bin/nmt_en_zh_transformer6x6_encoder_fp32.trt'
dec_init_engine='./model_bin/nmt_en_zh_transformer6x6_decoder_init_fp32.trt'
dec_engine='./model_bin/nmt_en_zh_transformer6x6_decoder_fp32.trt'
batch_size=2
beam_size=4
max_sequence_length=512
src_lang='en'
tgt_lang='zh'

# check output path
if [ ! -f "$tgt_save" ]; then
    touch "$tgt_save"
fi

python trt_infer.py \
    --src_file=${src_file} \
    --tgt_file=${tgt_file} \
    --tgt_save=${tgt_save} \
    --enc_engine=${enc_engine} \
    --dec_init_engine=${dec_init_engine} \
    --dec_engine=${dec_engine} \
    --batch_size=${batch_size} \
    --beam_size=${beam_size} \
    --max_sequence_length=${max_sequence_length} \
    --src_lang=${src_lang} \
    --tgt_lang=${tgt_lang}

