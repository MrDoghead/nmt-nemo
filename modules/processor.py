import re
from typing import List

import jieba
import opencc
from pangu import spacing
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

class ChineseProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities for Chinese.
    """

    def __init__(self):
        self.normalizer = opencc.OpenCC('t2s.json')

    def normalize(self, text: str) -> str:
        return self.normalizer.convert(text)

    def detokenize(self, text: List[str]) -> str:
        RE_WS_IN_FW = re.compile(
            r'([\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef])\s+(?=[\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef])'
        )

        detokenize = lambda s: spacing(RE_WS_IN_FW.sub(r'\1', s)).strip()
        return detokenize(' '.join(text))

    def tokenize(self, text: str) -> str:
        text = jieba.cut(text)
        return ' '.join(text)

class MosesProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities in Moses
    """

    def __init__(self, lang_id: str):
        self.moses_tokenizer = MosesTokenizer(lang=lang_id)
        self.moses_detokenizer = MosesDetokenizer(lang=lang_id)
        self.normalizer = MosesPunctNormalizer(lang=lang_id)

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenizes a list of tokens
        Args:
            tokens: list of strings as tokens
        Returns:
            detokenized string
        """
        return self.moses_detokenizer.detokenize(tokens)

    def tokenize(self, text: str):
        """
        Tokenizes text using Moses -> Sentencepiece.
        """
        return self.moses_tokenizer.tokenize(text, escape=False, return_str=True)

    def normalize(self, text: str):
        return self.normalizer.normalize(text)
