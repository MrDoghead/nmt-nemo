import torch
import numpy as np
from argparse import ArgumentParser
import logging

from modules.processor import ChineseProcessor, MosesProcessor
from modules.youtokentome_tokenizer import YouTokenToMeTokenizer
from modules.trt_helper import EncWrapper, DecWrapper
from modules.generator import BeamSearchGenerator

VOCAB_SIZE = 32000
HIDDEN_SIZE = 1024
MAX_SEQ_LENGTH = 512
device = torch.device('cpu')

encoder_tokenizer = YouTokenToMeTokenizer(
        model_path='./model_bin/tokenizer.encoder.32000.BPE.model',
        bpe_dropout=0.0,
        legacy=False)
decoder_tokenizer = YouTokenToMeTokenizer(
        model_path='./model_bin/tokenizer.decoder.32000.BPE.model',
        bpe_dropout=0.0,
        legacy=False)

enc_trt_path = './model_bin/nmt_en_zh_transformer6x6_encoder_fp32.trt'
dec_trt_path = './model_bin/nmt_en_zh_transformer6x6_decoder_fp32.trt'

def setup_processor(src_lang, tgt_lang):
    src_processor, tgt_processor = None, None
    if src_lang == 'zh':
        src_processor = ChineseProcessor()
    if src_lang is not None and src_lang not in ['ja', 'zh']:
        src_processor = MosesProcessor(src_lang)
    if tgt_lang == 'zh':
        tgt_processor = ChineseProcessor()
    if tgt_lang is not None and tgt_lang not in ['ja', 'zh']:
        tgt_processor = MosesProcessor(tgt_lang)
    return src_processor, tgt_processor

def batch_translate(src_ids, src_mask, args):
    batch_size = args.batch_size
    seq_len = src_ids.shape[1]
    shape_of_output = (batch_size, seq_len, HIDDEN_SIZE)
    # trt encoder engine
    enc_wrapper = EncWrapper(trt_path=enc_trt_path)
    enc_outputs = enc_wrapper.do_inference(
            src_ids,
            src_mask,
            shape_of_output,
            batch_size)
    src_hiddens = enc_outputs[0].reshape(shape_of_output)
    # convert numpy back to tensor
    encoder_hidden_states = torch.from_numpy(src_hiddens).to(device)
    encoder_input_mask = torch.from_numpy(src_mask).to(device)

    beam_search = BeamSearchGenerator(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            max_sequence_length=MAX_SEQ_LENGTH,
            dec_trt_path,
            pad=decoder_tokenizer.pad_id,
            bos=decoder_tokenizer.bos_id,
            eos=decoder_tokenizer.eos_id,
            max_sequence_length=args.max_sequence_length,
            max_delta_length=args.max_delta_length,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            len_pen=args.len_pen)
    beam_results = beam_search.forward(encoder_hidden_states=encoder_hidden_states, encoder_input_mask=encoder_input_mask)

    sys.exit()
    beam_results = filter_predicted_ids(beam_results)

    translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
    inputs = [self.encoder_tokenizer.ids_to_text(inp) for inp in src.cpu().numpy()]
    if self.target_processor is not None:
        translations = [
            self.target_processor.detokenize(translation.split(' ')) for translation in translations
        ]

    if self.source_processor is not None:
        inputs = [self.source_processor.detokenize(item.split(' ')) for item in inputs]

    return inputs, translations


def translate(text, args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_processor, tgt_processor = setup_processor(src_lang, tgt_lang)

    inputs = []
    for txt in text:
        print(txt)
        if src_processor is not None:
            txt = src_processor.normalize(txt)
            txt = src_processor.tokenize(txt)
        ids = encoder_tokenizer.text_to_ids(txt)
        ids = [encoder_tokenizer.bos_id] + ids + [encoder_tokenizer.eos_id]
        inputs.append(ids)
    max_len = max(len(txt) for txt in inputs)
    src_ids = np.ones((len(inputs), max_len), dtype=np.int32) * encoder_tokenizer.pad_id
    for i, txt in enumerate(inputs):
        src_ids[i][: len(txt)] = txt
    src_mask = np.array((src_ids != encoder_tokenizer.pad_id),dtype=np.int32)
    _, translations = batch_translate(src_ids, src_mask, args)

    return translations

def main():
    parser = ArgumentParser()
    parser.add_argument("--src_file", help="the source text file to translate", type=str)
    parser.add_argument("--tgt_file", help="the corresponding target text file", type=str)
    parser.add_argument("--tgt_save", help="the saving path of translated target texts", type=str)
    parser.add_argument("--enc_engine", help="the encoder trt engine", type=str)
    parser.add_argument("--dec_init_engine", help="the init decoder trt engine with none cached memory", type=str)
    parser.add_argument("--dec_engine", help="the main decoder trt engine with cached memory", type=str)
    parser.add_argument("--batch_size", type=int, default=256, help="inference batch size")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="max generation length")
    parser.add_argument("--beam_size", type=int, default=4, help="beam search size")
    parser.add_argument("--len_pen", type=float, default=0.6, help="beam search length penalty")
    parser.add_argument("--max_delta_length", type=int, default=5, help="the max length that tgt seq can longer than src seq")
    parser.add_argument("--src_lang", type=str, default='en', help="source language")
    parser.add_argument("--tgt_lang", type=str, default='zh', help="target language")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Translating: {args.src_file}")
    torch.set_grad_enabled(False)

    src_text = []
    pred_text = []
    count = 0
    with open(args.src_file, 'r') as src_f:
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                res = translate(src_text, args)
                sys.exit()
                pred_text += res
                src_text = []
            count += 1
            if count != 0 and count % 300 == 0:
               print(f"Translated {count} sentences")
        if len(src_text) > 0:
            pred_text += translate(src_text, args)

    with open(args.tgt_save, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")

if __name__=='__main__':
    main()

