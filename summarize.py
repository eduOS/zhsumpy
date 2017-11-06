#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from zhsumpy.modules.opts.sentence_extractor import TextRank
from zhsumpy.modules.opts.keywords_extractor import TFIDFExtractor
# from zhsumpy.modules import EmbeddingEvaluator as WordEvaluator
# from zhsumpy.modules import EmbeddingChecker as SimChecker
from zhsumpy.modules.similarity_sorter import SentenceSorter
# from zhsumpy.modules.opts.sentence_similarity_checker import TFIDFChecker
from cntk.tokenizer import JiebaTokenizer


class Summarizer(object):

    def __init__(self, max_sen=3, max_can=5):
        """
        input:
            max_an: the maxmum number of answers which should be kept
        """
        self._max_sen = max_sen
        self._max_can = max_can
        self._textrank = TextRank(
            keywords_extractor=TFIDFExtractor(True))
        self._tokenizer = JiebaTokenizer()
        self._sentence_sorter = SentenceSorter()

    def summarize(self, sentences, query=None):
        """
        input:
            sentences: a list of sentences
            query: if not None it the title or question
        return:
            the sorted sentences
        """
        if query:
            query = self._tokenizer.sentence2words(query, False)

        sentences, corpus = self._tokenizer.sentences2words(sentences)

        if len(sentences) <= self._max_sen:
            return sentences
        elif len(sentences) > self._max_can and query:
            corpus, sentences = self._sentence_sorter(
                query, corpus, sentences, self._max_sen)

        scores = self._textrank(corpus)
        if scores:
            try:
                return [
                    sentences[idx]
                    for (idx, sc) in scores.get_top_idx(self._max_sen)
                ]
            except Exception as e:
                print(e)
                return None
