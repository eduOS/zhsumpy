#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np


class Evaluator(object):
    """
    evaluate words in the 2D list
    input:
        a 2D list of words
    output:
        a 2D array of scores of the shape
    """

    def __call__(self):
        """
        """
        raise NotImplementedError("Abstract method")

    @property
    def scores(self):
        """
        the result array containing the scores of each word in order of the
        vocabulary
        """
        raise NotImplementedError("Abstract method")


class EmbeddingEvaluator(Evaluator):
    """
    input:
        can be only a 1D list, a single word is acceptable
    output:
        a matrix with one more dimention as the embedding
    """
    def __init__(self):
        from zhsumpy.data import embeddings
        self._embeddings = embeddings
        if self._embeddings is None:
            raise Exception("embeddings.pkl not found under ../data/")

        from zhsumpy.data import vocabulary
        self._dictionary = vocabulary
        if self._dictionary is None:
            raise Exception("vocabulary.pkl not found under ../data/.")

        if len(self._dictionary) != len(self._embeddings):
            raise Exception("The dictionary and the embeddings don't match..")

        self._word_embeddings = None

    def _get_word_embedding(self, word):
        """
        given a word return the embedding
        """
        embedding = np.zeros(len(self._dictionary))
        try:
            embedding = self._embeddings[self._dictionary[word]]
        except KeyError:
            embedding = self._embeddings[self._dictionary["UNK"]]
        return list(embedding)

    def get_embedding(self, word_list):
        if isinstance(word_list, unicode):
            word_list = [word_list]
        # assure that it is a 1D list rather than nested list
        try:
            assert isinstance(word_list[0], unicode)
        except IndexError:
            # the list is a blank list
            return 0
        em_list = []
        for word in word_list:
            em_list.append(self._get_word_embedding(word))
            # because of the length of words of each sentence is not the same
            # the lower dimention will be list
        self._word_embeddings = np.array(em_list)
        return self._word_embeddings

    @property
    def scores(self):
        return self._word_embeddings

    def __call__(self, word_list):
        """
        input:
            idx: the index of the matrix which can be an int or a list of ints
        """
        return self.get_embedding(word_list)
