__author__ = 'dsbatista'

import numpy as np


class Pattern(object):

    def __init__(self, t=None):
        self.confidence = 0
        self.tuples = set()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)

    def merge_patterns(self):
        for t in self.tuples:
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def sum(self):
        self.merge_patterns()
        vector = np.zeros(200)
        for t in self.patterns_words:
            print t
