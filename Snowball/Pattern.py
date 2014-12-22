__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Pattern(object):

    def __init__(self, t=None):
        self.positive = 0
        self.negative = 0
        self.confidence = 0
        self.tuples = set()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def __str__(self):
        return " | ".join([p for p in self.patterns_words]).encode("utf8")

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence(self):
        if self.positive or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
        self.update_confidence()
