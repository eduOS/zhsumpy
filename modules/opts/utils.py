#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import functools


def not_none(func):
    """
    decorator to make sure the sentence to be processed is not none
    this only validated on the middle level(utilization level)
    (rather thant on the constant or application level)
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._corpus:
            raise Exception("The corpus is None in %s" % func.__name__)
        else:
            try:
                func(self, *args, **kwargs)
            except TypeError:
                raise
    return wrapper
