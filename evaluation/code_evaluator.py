import ast

import astor
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from NL2code.evaluation import tokenize_for_bleu_eval
from NL2code.lang.py.parse import tokenize_code, de_canonicalize_code
from current_net_conf import *


class CodeEvaluator:

    def __init__(self):
        self.sample_accuracy = {}
        self.sample_bleu = {}
        self.sm = SmoothingFunction()

    def eval(self, generated_code, example):
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        accuracy = self._calculate_accuracy(generated_code, refer_source)
        bleu = self._calculate_bleu(generated_code, example)

        self.sample_accuracy[example.raw_id] = accuracy
        self.sample_bleu[example.raw_id] = bleu

        return accuracy, bleu

    def normalize_code(self, generated_code, example):
        normalized_code = de_canonicalize_code(generated_code, example.meta_data['raw_code'])
        for literal, place_holder in example.meta_data['str_map'].items():
            quoted_pc = '\'' + place_holder + '\''
            if quoted_pc in normalized_code:
                normalized_code = normalized_code.replace(quoted_pc, literal)
            if place_holder in normalized_code:
                normalized_code = normalized_code.replace(place_holder, literal)
        return normalized_code

    def _calculate_accuracy(self, generated_code, refer_source):
        refer_tokens = tokenize_code(refer_source)
        predict_tokens = tokenize_code(generated_code)
        return refer_tokens == predict_tokens

    def _calculate_bleu(self, generated_code, example):
        if DATA_SET_TYPE == DJANGO_DATA_SET_TYPE:
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = self.normalize_code(generated_code, example)

        elif DATA_SET_TYPE == HS_DATA_SET_TYPE:
            ref_code_for_bleu = example.code
            pred_code_for_bleu = generated_code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights,
                                   smoothing_function=self.sm.method3)
        return bleu_score

    def get_accuracy(self):
        return np.mean(self.sample_accuracy.values())

    def get_bleu(self):
        return np.mean(self.sample_bleu.values())
