import torch
import numpy as np
import argparse 
import logging
import time
from sacrebleu import corpus_bleu
import tensorrt as trt
import sys
sys.path.append('./')

from modules.processor import ChineseProcessor, MosesProcessor
from modules.youtokentome_tokenizer import YouTokenToMeTokenizer
from modules.trt_helper import EncWrapper
from modules.generator import BeamSearchGenerator
from modules.metrics import eval_bleu
from utils.infer_utils import MeasureTime

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

    def batch_translate(self, src_ids, src_mask, args, measurements):
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
        with MeasureTime(measurements, "encoder_time"):
            self.encoder.run_trt_engine(encoder_tensors)

        with MeasureTime(measurements, "generator_time"):
            gen_results = self.generator.forward(encoder_hidden_states=encoder_hidden_states, encoder_mask=encoder_mask)

        # postprocess
        with MeasureTime(measurements, "postprocess_time"):
            gen_results = self.filter_predicted_ids(gen_results)
            translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in gen_results.cpu().numpy()]
            inputs = [self.encoder_tokenizer.ids_to_text(inp) for inp in src_ids]
            if self.tgt_processor is not None:
                translations = [
                    self.tgt_processor.detokenize(translation.split(' ')) for translation in translations]
            if self.src_processor is not None:
                inputs = [self.src_processor.detokenize(item.split(' ')) for item in inputs]

        return inputs, translations

    def translate(self, text, args, measurements):
        # print(f'\ntranslating text: {text}')

        # preprocess
        with MeasureTime(measurements, "preprocess_time"):
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

        # translate this batch
        _, translations = self.batch_translate(src_ids, src_mask, args, measurements)
        # print('Translations:',translations)

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
        description='TensorRT NMT Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # initialize CUDA state
    torch.cuda.init()

    logging.info(f"Translating from: {args.src_file}")
    translator = Translator(args)
    measurements = {}

    src_text = []
    pred_text = []
    count = 0
    with open(args.src_file, 'r') as src_f, \
            torch.no_grad(), MeasureTime(measurements, "total_latency"):
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                with MeasureTime(measurements, "batch_latency"):
                    res = translator.translate(src_text, args, measurements)
                pred_text += res
                src_text = []
            count += 1
            if count != 0 and count % 100 == 0:
               print(f"{count} sentences translated...")
        if len(src_text) > 0:
            pred_text += translator.translate(src_text, args)

    # report measure time 
    for name in measurements:
        value = np.mean(measurements[name])
        if name == "total_latency":
            logging.info(f"{name}: {value}s")
        else:
            logging.info(f"Averaged {name}: {value}s per batch")

    # save predictions
    with open(args.tgt_save, 'w') as tgt_f:
        for line in pred_text:
            tgt_f.write(line + "\n")
    logging.info(f'Target translations saved at {args.tgt_save}')

    # eval
    eval_bleu(prediction_file=args.tgt_save, 
            ground_truth_file=args.tgt_file,
            batch_size=args.batch_size,
            lang=args.tgt_lang)


if __name__=='__main__':
    main()

