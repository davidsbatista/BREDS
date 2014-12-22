# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef = _before
            self.bet = _between
            self.aft = _after
            self.patterns_vectors = list()
            self.patterns_words = list()

        def __str__(self):
            return str(self.patterns_words).encode("utf8")

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0


        def __init__(self):
            pass