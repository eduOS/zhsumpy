#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import warnings

from zhsumpy.modules.opts.keywords_extractor import TFIDFExtractor
from zhsumpy.modules.opts.sentence_similarity_checker import EmbeddingChecker
from zhsumpy.modules.opts.words_evaluator import EmbeddingEvaluator
from zhsumpy.modules.opts.sentence_evaluator import SentenceEmbedding
from cntk.tokenizer import JiebaTokenizer


class Sorter(object):
    pass


class SentenceSorter(Sorter):
    def __init__(self,
                 tokenizer=JiebaTokenizer(),
                 keywords_extractor=TFIDFExtractor(True),
                 # this should be optional
                 words_evaluator=EmbeddingEvaluator(),
                 sentence_evaluator=SentenceEmbedding(),
                 sentence_similarity_checker=EmbeddingChecker(),
                 ):
        """
        1. receive the lists of words
        2. find the key words
        3. calculate the sentence embedding
        4. calculate the similarity of each candidates to the target
        parameters are all classes, since the init is just for design
        not initiation

        return:
            the target and the sorted sentences

        TODO: other algorithms should be integrated
        """
        self._target = None
        self._corpus = None
        self._candidates = None
        self._extractor = keywords_extractor
        self._words_evaluator = words_evaluator
        self._sentence_evaluator = sentence_evaluator
        self._checker = sentence_similarity_checker
        self._tokenizer = tokenizer
        self._similarities = None

    def _prepare(self):
        """
        tokenize the sentences, extract keywords for each sentence and transform
        keywords into embeddings
        """
        # target = self._tokenizer.sentence2words(
        #     sentence=self._target,
        #     stopwords=False
        # )
        # if target == []:
        #     return None, None
        # self._candidates, corpus = self._tokenizer.sentences2words(
        #     sentences=self._candidates,
        #     stopwords=False,
        # )
        assert self._corpus is not None, "corpus is None"
        keywords_matrix = self._extractor(
            self._corpus
        ).get_keywords(percent=0.8)
        candidate_scores = []
        for row in keywords_matrix:
            if row != []:
                candidate_scores.append(self._words_evaluator(row))
            else:
                candidate_scores.append(None)

        target_scores = self._words_evaluator(self._target)

        return target_scores, candidate_scores

    def _check_similarity(self):
        target_scores, candidate_scores = self._prepare()
        self._similarities = [
            self._checker(candidate, target_scores)
            for candidate in candidate_scores
        ]
        try:
            assert len(self._similarities) == len(self._corpus)
        except:
            if self._corpus and self._similarities:
                warnings.warn(
                    "The length %s of the corpus "
                    "is not equal to that %s of the similiraties calculated."
                    % (
                        unicode(len(self._corpus)), # NOQA
                        unicode(len(self._similarities)) # NOQA
                    )
                )
            return None
        sorted_corpus = [
            t[1] for t in sorted(
                zip(self._similarities, self._corpus))][::-1]
        return sorted_corpus

    def _sort_candidates(self):
        assert len(self._candidates) == len(self._corpus)
        if self._candidates:
            sorted_candidates = [
                t[1] for t in sorted(
                    zip(self._similarities, self._candidates))][::-1]
            return sorted_candidates
        else:
            return None

    def __call__(self, target, corpus=None, candidates=None, top=None):
        """
        corpus: the word list of the candidates
        candidates: the sentences of the candidates
        """
        if corpus is None:
            self._target = target[0]
            self._corpus = target[1:]
        else:
            self._target = target
            self._corpus = corpus
        self._candidates = candidates

        sorted_corpus = self._check_similarity()
        if not sorted_corpus:
            return (None, None)
        # what if sorted_corpus is None?

        top = len(corpus) - 1 if not top else top

        if candidates:
            sorted_candidates = self._sort_candidates()
            return sorted_corpus[:top], sorted_candidates[:top]

        return sorted_corpus[:top]
