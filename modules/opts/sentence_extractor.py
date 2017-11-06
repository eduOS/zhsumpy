#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import networkx as nx
import numpy as np

from zhsumpy.modules.opts.words_evaluator import EmbeddingEvaluator
from zhsumpy.modules.opts.sentence_similarity_checker import EmbeddingChecker
from zhsumpy.modules.opts.keywords_extractor import TFIDFExtractor


class Extractor(object):

    @property
    def score(self):
        """
        the property to get the result score
        """
        raise NotImplementedError("Abstract method")

    def __call__(self, corpus):
        raise NotImplementedError("Abstract method")


class TextRank(Extractor):

    def __init__(
        self,
        words_evaluator=EmbeddingEvaluator(),
        sentence_similarity_checker=EmbeddingChecker(),
        keywords_extractor=None,
        alpha=0.85
    ):
        """
        inputs:
            keywords_extractor: a class to score keywords

            sentence_similarity_checker: an class to calculate similarity
            between two sentences
        """
        self._alpha = alpha
        self._words_evaluator = words_evaluator
        self._similarity_checker = sentence_similarity_checker
        if keywords_extractor is None:
            self._keywords_extractor = TFIDFExtractor()
        elif keywords_extractor:
            self._keywords_extractor = keywords_extractor
        else:
            self._keywords_extractor = False
        self._score = None

    def _get_graph(self, corpus):
        """
        The graph is undirected since the difference between the results of
        directed and undirected graph is very small

        input:

        return:
            the graph
        """
        corpus_length = len(corpus)
        graph = np.zeros((corpus_length, corpus_length))

        for idx_i in range(corpus_length):
            score_i = self._words_evaluator(corpus[idx_i])
            for idx_j in range(idx_i, corpus_length):
                score_j = self._words_evaluator(corpus[idx_j])
                score = self._similarity_checker(score_i, score_j)
                graph[idx_i, idx_j] = score
                graph[idx_j, idx_i] = score
        nx_graph = nx.from_numpy_matrix(graph)
        return nx_graph

    def __call__(
        self,
        corpus,
    ):
        """
        inputs:
            corpus: a 2D list with sentences(also lists in 0 axis) each
            containing words which should be considered. No empty list should
            be contained
        return:
            return the instance from which the score property can be got
            The score would be None if
        Exception:
            if empty list contained
            pagerank: power iteration failed to converge in 100 iterations.
        """
        assert corpus is not None
        if self._keywords_extractor:
            corpus = self._keywords_extractor(corpus).get_keywords(percent=0.8)

        nx_graph = self._get_graph(corpus)
        try:
            score = nx.pagerank(nx_graph, self._alpha)
        except nx.NetworkXError:
            print("Oops.. The pagerank failed..")
            print("Here is the corpus: %s" % unicode(corpus))
            print("The corresponding Chinese.")
            for row in corpus:
                print(", ".join(row))
            print("Most probably the graph is None: %s" % unicode(nx_graph))
            return None
        self._score = score
        return self

    @property
    def score(self):
        return self._score

    def get_top_idx(self, top=1):
        return [(idx, sc) for (idx, sc) in sorted(
            self._score.items(), key=lambda x:x[1], reverse=True)][:top]

    def get_through_scale(self, scale):
        highest = max(self._score.values())
        lowest = min(self._score.values())
        # it is intuitive
        score = highest - (highest - lowest) * scale
        return [(idx, sc) for (idx, sc) in sorted(
            self._score.items(),
            key=lambda x:x[1], reverse=True) if sc >= score]


class LuhnExtractor(Extractor):

    def luhns_method(sentences, important_words):
        """
        output:
            get the the index and the max score of every sentence
            according to the important words clusters in it
        """
        score = []
        sentence_idx = -1

        #  !this should be replaced
        # [for wd in jieba.cut(''.join(s), cut_all=False) for s in sentences]:
        for s in sentences:
            sentence_idx += 1
            word_idx = []

            # For each word in the word list...
            for w in important_words:
                try:
                    # Compute an index for where any important words occur in
                    # the sentence

                    # mark the important words in the sentence
                    word_idx.append(s.index(w))
                except:  # w not in this particular sentence
                    pass

            word_idx.sort()

            # It is possible that some sentences may not contain any important
            # words at all
            if len(word_idx) == 0:
                continue

            # Using the word index, compute clusters by using a max distance
            # threshold for any two consecutive words

            clusters = []
            cluster = [word_idx[0]]
            i = 1
            # if important words appear intensively somewhere keep their index(a
            # cluster) as an item in the clusters list, if not only one word_idx
            # will be in the cluster
            while i < len(word_idx):
                CLUSTER_THRESHOLD = 5
                if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)

            # Score each cluster. The max score for any given cluster is the
            # score for the sentence

            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                # the difference between the above two value: the former is the
                # number of significant words; the later however is the index
                # distance between the last important word and the first one the
                # score can be treated as calculating the density of important
                # words in the sentence
                score = 1.0 * significant_words_in_cluster \
                    * significant_words_in_cluster / total_words_in_cluster

                if score > max_cluster_score:
                    max_cluster_score = score

            score.append((sentence_idx, max_cluster_score))

        return score
