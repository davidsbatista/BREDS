__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros
from Word2VecWrapper import Word2VecWrapper


class Pattern(object):

    def __init__(self, t=None):
        self.single_vector = zeros(200)
        self.confidence = 0
        self.tuples = set()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)
            for p in t.patterns_words:
                self.patterns_words.add(p)

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
