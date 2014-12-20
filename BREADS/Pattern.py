__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros
from Word2VecWrapper import Word2VecWrapper


class Pattern(object):

    def __init__(self, t=None):
        self.single_vector = zeros(200)
        self.positive = 0
        self.negative = 0
        self.confidence = 0
        self.tuples = set()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def update_confidence(self):
        if self.positive or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)

    def merge_patterns(self):
        for t in self.tuples:
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def calculate_single_vector(self):
        self.merge_patterns()
        pattern_vector = zeros(200)
        for p in self.patterns_words:
            vector_p = Word2VecWrapper.pattern2vector(p, 200)
            pattern_vector += vector_p
        self.single_vector = pattern_vector

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
        self.update_confidence()

    def __str__(self):
        return " | ".join([p for p in self.patterns_words]).encode("utf8")
