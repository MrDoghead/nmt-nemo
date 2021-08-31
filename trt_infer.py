import torch
import numpy as np
import argparse 
import logging
import time
from sacrebleu import corpus_bleu
import tensorrt as trt

from modules.processor import ChineseProcessor, MosesProcessor
from modules.youtokentome_tokenizer import YouTokenToMeTokenizer
from modules.trt_helper import EncWrapper
from modules.generator import BeamSearchGenerator
from modules.metrics import eval_bleu

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logging.basicConfig(level=logging.INFO)

class Translator(object):
    def __init__(self,args):
        self.encoder_tokenizer = YouTokenToMeTokenizer(
                model_path=args.enc_tokenizer,
                bpe_dropout=0.0,
                legacy=False)
        self.decoder_tokenizer = YouTokenToMeTokenizer(
                model_path=args.dec_tokenizer,
                bpe_dropout=0.0,
                legacy=False)
        self.src_processor, self.tgt_processor = \
                self.setup_processor(args.src_lang, args.tgt_lang)
        self.encoder = EncWrapper(args.enc_engine)
        self.generator = BeamSearchGenerator(
                self.decoder_tokenizer.vocab_size,
                args.hidden_size,
                args.dec_init_engine,
                args.dec_engine,
                pad=self.decoder_tokenizer.pad_id,
                bos=self.decoder_tokenizer.bos_id,
                eos=self.decoder_tokenizer.eos_id,
                max_sequence_length=args.max_sequence_length,
                max_delta_length=args.max_delta_length,
                batch_size=args.batch_size,
                beam_size=args.beam_size,
                len_pen=args.len_pen,
                chk_path=args.checkpoint)

    def filter_predicted_ids(self, ids):
        ids[ids >= self.decoder_tokenizer.vocab_size] = self.decoder_tokenizer.unk_id
        return ids

    def setup_processor(self, src_lang, tgt_lang):
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

    def batch_translate(self, src_ids, src_mask, args):
        batch_size, seq_len = src_ids.shape
        shape_of_output = (batch_size, seq_len, args.hidden_size)
        # move inputs from cpu to gpu
        input_ids = torch.from_numpy(src_ids).cuda()
        encoder_mask = torch.from_numpy(src_mask).cuda()
        encoder_hidden_states = torch.zeros(shape_of_output).cuda()

        encoder_tensors = {
                "inputs":
                {"input_ids": input_ids, "encoder_mask": encoder_mask},
                "outputs":
                {"last_hidden_states": encoder_hidden_states}}
        t1=time.perf_counter()
        self.encoder.run_trt_engine(encoder_tensors)
        t2=time.perf_counter()
        logging.info(f'encode time: {t2-t1}s')

        t3=time.perf_counter()
        gen_results = self.generator.forward(encoder_hidden_states=encoder_hidden_states, encoder_mask=encoder_mask)
        t4=time.perf_counter()
        logging.info(f'beamsearch time: {t4-t3}s')

        # postprocess
        t5=time.perf_counter()
        gen_results = self.filter_predicted_ids(gen_results)
        translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in gen_results.cpu().numpy()]
        inputs = [self.encoder_tokenizer.ids_to_text(inp) for inp in src_ids]
        if self.tgt_processor is not None:
            translations = [
                self.tgt_processor.detokenize(translation.split(' ')) for translation in translations]
        if self.src_processor is not None:
            inputs = [self.src_processor.detokenize(item.split(' ')) for item in inputs]
        t6=time.perf_counter()
        logging.info(f'postprocess time: {t6-t5}')

        return inputs, translations

    def translate(self, text, args):
        logging.info(f'\ntranslating text: {text}')

        # preprocess
        t1=time.perf_counter()
        inputs = []
        for txt in text:
            if self.src_processor is not None:
                txt = self.src_processor.normalize(txt)
                txt = self.src_processor.tokenize(txt)
            ids = self.encoder_tokenizer.text_to_ids(txt)
            ids = [self.encoder_tokenizer.bos_id] + ids + [self.encoder_tokenizer.eos_id]
            inputs.append(ids)
        max_len = max(len(txt) for txt in inputs)
        src_ids = np.ones((len(inputs), max_len), dtype=np.int32) * self.encoder_tokenizer.pad_id
        for i, txt in enumerate(inputs):
            src_ids[i][: len(txt)] = txt
        src_mask = np.array((src_ids != self.encoder_tokenizer.pad_id),dtype=np.int32)
        t2=time.perf_counter()
        logging.info(f'preprocess time: {t2-t1}')

        # translate this batch
        _, translations = self.batch_translate(src_ids, src_mask, args)
        logging.info('Translations:',translations)

        return translations

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument("--src_file", help="the source text file to translate", type=str)
    parser.add_argument("--tgt_file", help="the corresponding target text file", type=str)
    parser.add_argument("--tgt_save", help="the saving path of translated target texts", type=str)
    parser.add_argument("--enc_engine", help="the encoder trt engine", type=str)
    parser.add_argument("--dec_init_engine", help="the init decoder trt engine with none cached memory", type=str)
    parser.add_argument("--dec_engine", help="the main decoder trt engine with cached memory", type=str)
    parser.add_argument("--enc_tokenizer", help="encoder tokenizer", type=str)
    parser.add_argument("--dec_tokenizer", help="decoder tokenizer", type=str)
    parser.add_argument("--batch_size", type=int, default=8, help="inference batch size")
    parser.add_argument("--hidden_size", type=int, default=512, help="the encoder and decoder hidden size")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="max generation length")
    parser.add_argument("--beam_size", type=int, default=4, help="beam search size")
    parser.add_argument("--len_pen", type=float, default=0.6, help="beam search length penalty")
    parser.add_argument("--max_delta_length", type=int, default=5, help="the max length that tgt seq can longer than src seq")
    parser.add_argument("--src_lang", type=str, default='en', help="source language")
    parser.add_argument("--tgt_lang", type=str, default='zh', help="target language")
    parser.add_argument("--checkpoint", type=str, help="the path to restore checkpoint")

    return parser

def main():
    parser = argparse.ArgumentParser(
        description='NMT-trt Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # initialize CUDA state
    torch.cuda.init()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Translating from: {args.src_file}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    translator = Translator(args)

    src_text = []
    pred_text = []
    count = 0
    with open(args.src_file, 'r') as src_f, torch.no_grad():
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                t1 = time.perf_counter()
                res = translator.translate(src_text, args)
                t2 = time.perf_counter()
                logging.info(f'bs={args.batch_size} cost {t2-t1}s')
                pred_text += res
                src_text = []
            count += 1
            if count != 0 and count % 100 == 0:
               logging.info(f"{count} sentences translated")
        if len(src_text) > 0:
            pred_text += translator.translate(src_text, args)

    with open(args.tgt_save, 'w') as tgt_f:
        for line in pred_text:
            tgt_f.write(line + "\n")
    logging.info(f'Target translations saved at {args.tgt_save}')

    # eval
    eval_bleu(args.tgt_save,args.tgt_file,args.batch_size,args.tgt_lang)


if __name__=='__main__':
    main()

