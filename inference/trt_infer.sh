#!/bin/sh

src_file='./data/test_1k.en'
tgt_file='./data/test_1k.zh'
tgt_save='./data/pred_trt_fp32.zh'
enc_engine='./model_bin/nmt_en_zh_transformer6x6_encoder_fp32.engine'
dec_init_engine='./model_bin/nmt_en_zh_transformer6x6_decoder_init_fp32.engine'
dec_engine='./model_bin/nmt_en_zh_transformer6x6_decoder_fp32.engine'
enc_tokenizer='./model_bin/tokenizer.encoder.32000.BPE.model'
dec_tokenizer='./model_bin/tokenizer.decoder.32000.BPE.model'
batch_size=8
hidden_size=1024
beam_size=4
max_sequence_length=512
src_lang='en'
tgt_lang='zh'
chk_path='./model_bin/model_weights.ckpt'

# check output path
if [ ! -f "$tgt_save" ]; then
    touch "$tgt_save"
fi

python inference/trt_infer.py \
    --src_file=${src_file} \
    --tgt_file=${tgt_file} \
    --tgt_save=${tgt_save} \
    --enc_engine=${enc_engine} \
    --dec_init_engine=${dec_init_engine} \
    --dec_engine=${dec_engine} \
    --enc_tokenizer=${enc_tokenizer} \
    --dec_tokenizer=${dec_tokenizer} \
    --batch_size=${batch_size} \
    --hidden_size=${hidden_size} \
    --beam_size=${beam_size} \
    --max_sequence_length=${max_sequence_length} \
    --src_lang=${src_lang} \
    --tgt_lang=${tgt_lang} \
    --checkpoint=${chk_path}

