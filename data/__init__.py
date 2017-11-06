#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import pickle
from codecs import open
from os.path import join, dirname, abspath

path = dirname(abspath(__file__))

try:
    embeddings = pickle.load(open(join(path, 'embeddings.pkl')))
except:
    embeddings = None
try:
    vocabulary = pickle.load(open(join(path, 'vocabulary.pkl')))
except:
    vocabulary = None
