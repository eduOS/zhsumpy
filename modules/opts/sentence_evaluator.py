#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np


class Evaluator(object):
    def __init__(self):
        self._score = None

    @property
    def score(self):
        """
        the property to get the result score
        """
        raise NotImplementedError("Abstract method")


class SentenceEmbedding(Evaluator):
    """
    later more complex algorithms will be implemented
    https://www.aclweb.org/anthology/P/P16/P16-1048.pdf
    """
    def __init__(self):
        self._embedding = None

    @property
    def embedding(self):
        return self._embedding

    def mean_score(self, vector):
        if isinstance(vector, list):
            embeddings_array = np.array(vector)
        else:
            embeddings_array = vector
        if isinstance(embeddings_array, np.ndarray):
            self._embedding = np.mean(embeddings_array, axis=0)

        # for the case that vector is []
        if np.isnan(self._embedding).any():
            # this will cause bugs afterwards
            self._embedding = None
        return self._embedding
