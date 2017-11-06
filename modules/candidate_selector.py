#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import re

from cntk.constants import offals


class Selector():
    def __init__(self):
        '''
        the answer should be classified into several types, each type for
        different kinds of processing
        1. answers with only one paragraph but more than one sentence
            This kind of answers should be extracted using submodular
        2. answers with more than one paragraph and each with more
        than one sentence
            This kind of answers should be also extracted using submodular
        3. answers with points listed
            This kind of answers can be processed easily by just choose one
            randomly from all points
        4. answers with pictures
            If too many pictures(pic num bigger than 3) are included then ignore
        5. answers with only too long sentences
            delete
        '''
        self._candidates = None

    def classify(self):
        """
        1. each paragraph has only one sentence
        2. only one paragraph
        """
        if self._candidates.replace('\n', ''):
            pass

    def from_points(self):
        """
        the process is the answer with bullets in it like firsly and secondly
        and etc
        sentences in bold
        """
        bullets = offals.BULLET
        if re.search(self._candidates, bullets):
            pass

    def set_candidates(self, candidates):
        self._candidates = candidates

    def from_story(self):
        """
        if the text is too long and contain some comment tags in it
        """
        pass

    def __call__(self, candidates):
        self._candidates = candidates
