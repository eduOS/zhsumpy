#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from itertools import chain
from operator import itemgetter
import networkx as nx
import warnings
import math

from cntk.constants import stopwords
from cntk.tokenizer import JiebaTokenizer


class Extractor(object):
    """
    evaluate words in the 2D list
    input:
        a 2D list of words

        Internally, all words are stored in the form of a two dimentional matrix
        the horizental axis is the vocabulary and the vertical axis is the
        corpus

    output:
        a 2D array of scores of the shape
    """

    def __init__(self):
        self._corpus = None
        # the raw input data
        self._vocabulary = None
        # the words set of the corpus
        self._score_matrix = None
        # the scores in the shape of (len(corpus), len(vocabulary))
        self._keywords = None
        # the extracted keywords

    def set_corpus(self, corpus):
        self._corpus = corpus

    def _generate_vocabulary(self):
        assert self._corpus is not None
        self._vocabulary = list(
            set([term for doc in self._corpus for term in doc]))
        # why array rather than just list?

    def _word2idx(self, term):
        if not self._vocabulary:
            self._generate_vocabulary()
        try:
            idx = self._vocabulary.index(term)
        except:
            raise Exception(
                "the corpus is %s,"
                "_vocabulary is %s,"
                "and the searched term is %s"
                % (
                    unicode(self._corpus),
                    unicode(self._vocabulary),
                    unicode(term)
                )
            )
        return idx

    def _idx2word(self, idx):
        if not self._vocabulary:
            self._generate_vocabulary()
        return self._vocabulary[idx]

    def _extract_keywords(self, order=True, top=None, percent=None):
        # this is different for textrank extractor
        if not top and not percent and self._corpus:
            warnings.warn(
                "The top and percent are both None,"
                "then all words returned will be kept the same"
            )
            return self._corpus
        sorted_score = np.sort(np.unique(self._score_matrix))[::-1]
        if percent:
            tmp = int(len(sorted_score) * percent)
            top = tmp if tmp > 0 else 1
        threshold = sorted_score[top-1]

        keywords = []
        # maybe the following block can be replaced by a numpy operation
        for row in self._score_matrix:
            words_row = []
            for idx, score in enumerate(row):
                if score > threshold:
                    words_row.append(self._idx2word(idx))
            # if no keywords in a sentence return a []
            keywords.append(words_row)

        if not order:
            self._keywords = list(set(list(chain.from_iterable(keywords))))
        else:
            self._keywords = keywords

    def get_keywords(self, percent=None, order=True, top=None):
        """
        if order is true return a 2D list of the shape
            (len(corpus), len(keywords each row))
        if not return a list of keywords

        percent: the top *percent* of keywords to extract
        """
        assert self._corpus is not None
        assert self._vocabulary is not None
        assert self._score_matrix is not None
        if not percent and not top and self._keywords:
            # just as for the property
            warnings.warn(
                "Percent and top are not set, return the last keyword or None"
            )
            return self._keywords
        self._extract_keywords(top=top, order=order, percent=percent)
        return self._keywords

    # keywords = property(get_keywords)

    # if I add an abstract function here TFIDFExtractor() will be still a class
    # but why?


class TFIDFExtractor(Extractor):
    """
    input:
        corpus: a 2D list with words of each sentence as each row
    return:
        a matrix of shape (sentences, vocabulary), the index can be
        changed to the corresponding word using id2word(index) if
        no index provided, otherwise return the vector of that row only
    """

    def __init__(self, reduce_sparseness=False):
        """
        reduce_sparseness is for reducing the sparseness of the vocabulary
        that is delete all items in any two vectors that are all zero in
        that dimention, it is not irreversable and hence you cannot use the
        scores to match the corresponding words. It is a lossy reduction
        """
        assert type(reduce_sparseness) is bool
        self._reduce_sparseness = reduce_sparseness
        super(TFIDFExtractor, self).__init__()

    def _tf(self, term, doc, normalize=True):
        # the times the term appears in the doc
        if normalize:
            try:
                return doc.count(term) / float(len(doc))
            except:
                return 0
        else:
            return doc.count(term) / 1.0

    def _idf(self, term):
        # the frequency the term appears in the whole corpus
        num_texts_with_term = len(
            [1 for doc in self._corpus if term in doc]
        )
        try:
            return 1.0 + math.log(len(self._corpus) / num_texts_with_term)
        except ZeroDivisionError:
            return 1.0

    def _tf_idf(self, term, doc):
        return self._tf(term, doc) * self._idf(term)

    # this function also seems useless
    def refresh(self):
        self._get_tfidf_matrix()

    def _get_tfidf_matrix(self):
        assert self._vocabulary is not None
        matrix = []
        for doc in self._corpus:
            row = [0] * len(self._vocabulary)
            for term in set(doc):
                row[self._word2idx(term)] = self._tf_idf(term, doc)
            matrix.append(row)

        assert matrix is not None

        self._score_matrix = np.array(matrix)

    def __call__(self, corpus, idx=None):
        """
        input:
            idx: the index of the matrix which can be an int or a list of ints
            corpus: if not provided use the one passed last time
            The first parameter must be corpus for consistency

        return:
            each item in the returned tf-idf matrix will be a
            tuple: (term, score)
        """
        assert corpus is not None
        self._corpus = corpus
        self._generate_vocabulary()
        self._get_tfidf_matrix()

        if isinstance(idx, list) and self._reduce_sparseness:
            array = self._score_matrix[idx]
            # see here for the explanation:
            # http://stackoverflow.com/a/40200097/3552975
            return array[:, array.any(0)]
        elif isinstance(idx, int) or isinstance(idx, list):
            # only look for some rows
            return self._score_matrix[idx]
        elif idx is None:
            return self


class TextRankExtractor(Extractor):
    """
    input:
        a string of Chinese words or a 2D list of words with each raw as
        sentence and each item as a word in the sentence
    output:
        only a list of keywords
    """
    def __init__(
        self,
        pos=['n', 'ns', 'vn', 'v'],
        window=2,
        alpha=0.85,
    ):
        super(TextRankExtractor, self).__init__()
        self._pos = pos
        self._window = window
        self._alpha = alpha
        # super(TextRankExtractor, self).__init__(corpus)

    def _have_corpus(self):
        if isinstance(self._corpus, unicode):
            self._corpus = JiebaTokenizer().text2words(self._text, pos=True)

    def _get_vertex(self):
        return [
            w.word for w in self._corpus if
            w.word not in stopwords and w.flag in self._pos
        ]

    def _generate_cooccur(self, window=2):
        word_list = list(chain.from_iterable(self._corpus))
        window = 2 if window < 2 else window
        # itertools.combination is more convenient

        for x in xrange(1, window):
            if x >= len(word_list):
                break
            word_list2 = word_list[x:]
            res = zip(word_list, word_list2)
            for r in res:
                yield r

    def _rank(self, top):
        # a percent parameter should also be provided
        vertex = self._get_vertex()
        graph = np.zeros((len(vertex), len(vertex)))
        for w1, w2 in self._generate_cooccur(self._window):
            if w1 in vertex and w2 in vertex:
                graph[self._word2idx(w1)][self._word2idx(w2)] = 1
                graph[self._word2idx(w1)][self._word2idx(w2)] = 1
                # should I add the above one?
                # what if I replace 1 with the occurance of the edge
        nx_graph = nx.from_numpy_matrix(graph)
        ranked_idx = nx.pagerank(nx_graph, self._alpha).items()
        idx_by_rank = sorted(ranked_idx, key=itemgetter(1), reverse=True)
        if top > 0 and top < 1:
            top = int(len(vertex) * top)
        elif isinstance(top, int) and top > 0 and top < len(vertex):
            pass
        else:
            raise Exception("The top should be either a number or a percentage")

        self._keywords = {
            self._idx2word(idx): sc for idx, sc in idx_by_rank[:top]}

    def __call__(self, corpus, top=.4):
        """
        corpus can not only be a 2D list of words with a tag for each word,
        but also be a string punctuations should be deleted
        top can be a integer or a percentage
        """
        self._corpus = corpus
        self._have_corpus()
        self._generate_vocabulary()
        self._rank(top)
