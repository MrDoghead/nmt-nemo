from sacrebleu import corpus_bleu
import logging
import numpy as np

def eval_bleu(prediction_file, ground_truth_file, batch_size, lang):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Evaluating using bleu score.")
    logging.info(f'Prediction_file: {prediction_file}')
    logging.info(f'Ground_truth_file: {ground_truth_file}')

    sb_score_list = []
    predictions = []
    ground_truths = []
    with open(ground_truth_file, 'r') as gt_f:
        with open(prediction_file, 'r') as pred_f:
            pred_texts = pred_f.readlines()
            for i,line in enumerate(gt_f):
                gt_text = line.strip()
                pred_text = pred_texts[i]
                if gt_text and pred_text:
                    predictions.append(pred_text)
                    ground_truths.append(gt_text)
                if len(ground_truths) == batch_size:
                    if lang in ['ja']:
                        sacre_bleu = corpus_bleu(predictions, [ground_truths], tokenize="ja-mecab")
                    elif lang in ['zh']:
                        sacre_bleu = corpus_bleu(predictions, [ground_truths], tokenize="zh")
                    else:
                        sacre_bleu = corpus_bleu(predictions, [ground_truths], tokenize="13a")
                    sb_score_list.append(sacre_bleu.score)
                    logging.info(f'{i} texts evaled and the bleu score is {sacre_bleu}.')
                    predictions = []
                    ground_truths = []

            avg_sb_score = np.mean(sb_score_list)
            logging.info(f'The final averaged bleu score is {avg_sb_score}.')

if __name__=='__main__':
    prediction_file = '../data/pred_nemo.zh'
    ground_truth_file = '../data/test_1k.zh'
    batch_size = 16
    lang = 'zh'
    eval_bleu(prediction_file, ground_truth_file, batch_size, lang)
