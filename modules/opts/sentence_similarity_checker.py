#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import math
import numpy as np

from zhsumpy.modules.opts.sentence_evaluator import SentenceEmbedding


class Checker(object):
    """
    check similarity between two sentences
    several methods are available
    reference:
        http://www.cis.drexel.edu/faculty/thu/research-papers/dawak-547.pdf
    input:
        two arrays
    output:
        a score representing the similarity of the two arrays
    """

    def __init__():
        raise NotImplementedError("Abstract method")

    def __call__(self, array1, array2):
        """
        """
        raise NotImplementedError("Abstract method")

    @property
    def score(self):
        """
        the property to get the result score
        """
        raise NotImplementedError("Abstract method")


class TFIDFChecker(Checker):

    def __init__(self):
        """
        inputs:
            a 2D array with two rows of scores for each word
        output:
            the semilarity score
        """
        self._score = None
        # for the tfidf checker the keyword evaluator is set beforehand

    def _get_score(self, array1, array2):
        try:
            return array1.dot(array2 / math.sqrt(sum(array1**2)*sum(array2**2)))
        except ZeroDivisionError:
            return 0

    @property
    def score(self):
        return self._score

    def __call__(self, array1, array2):
        self._get_score(array1, array2)
        return self._score


class EmbeddingChecker(Checker):
    """
    input:
        two arrays of word embeddings
    output:
        the similarity score between these two arrays
    """
    def __init__(
        self,
        sentence_embedding=SentenceEmbedding()
    ):
        self._sentence_embedding = sentence_embedding
        self._score = None

    def _get_score(self, array1, array2):
        # the sentence embedding can be changed to other methods
        average_embedding_1 = self._sentence_embedding.mean_score(array1)
        average_embedding_2 = self._sentence_embedding.mean_score(array2)
        if average_embedding_1 is None or average_embedding_2 is None:
            self._score = 0
        else:
            self._score = np.dot(
                average_embedding_1, average_embedding_2) / (
                    np.linalg.norm(average_embedding_1) * np.linalg.norm(
                        average_embedding_2))

    @property
    def score(self):
        return self._score

    def __call__(self, array1, array2):
        if type(array1) is int or type(array2) is int:
            # the array may be an int
            return 0
        self._get_score(array1, array2)
        return self._score


class LinguisticChecker(Checker):

    def by_linguistic(self):
        pass


class WordOverlapChecker(Checker):
    def __init__(self):
        self.word_list1 = None
        self.word_list2 = None

    def by_word_overlap(self):
        words = list(set(self.word_list1 + self.word_list2))
        vector1 = [float(self.word_list1.count(word)) for word in words]
        # a vector of word_list1 on the vocabulary
        vector2 = [float(self.word_list2.count(word)) for word in words]
        vector3 = [vector1[x]*vector2[x] for x in xrange(len(vector1))]
        # the dot multiplication
        vector4 = [1 for num in vector3 if num > 0.]
        # calculate the duplicated number, most of the items are 0
        co_occur_num = sum(vector4)

        if abs(co_occur_num) <= 1e-12:
            return 0.

        denominator = math.log(
            float(len(self.word_list1))) + math.log(float(len(self.word_list2)))

        if abs(denominator) < 1e-12:
            return 0.

        return co_occur_num / denominator

    def __call__(self, word_list1, word_list2):
        self.word_list1 = word_list1
        self.word_list2 = word_list2
        return self.by_word_overlap()


class LevenShteinChecker(Checker):
    def lDistance(self, firstString, secondString):
        """Function to find the Levenshtein distance between two words/sentences -
        gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python
        """
        if len(firstString) > len(secondString):
            firstString, secondString = secondString, firstString
        distances = range(len(firstString) + 1)
        for index2, char2 in enumerate(secondString):
            newDistances = [index2 + 1]
            for index1, char1 in enumerate(firstString):
                if char1 == char2:
                    newDistances.append(distances[index1])
                else:
                    newDistances.append(1 + min((distances[index1],
                                                distances[index1 + 1],
                                                newDistances[-1])))
            distances = newDistances
        return distances[-1]

    def __call__(self, string1, string2):
        return self.lDistance(string1, string2)
