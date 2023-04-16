import uuid

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


class Pattern(object):
    def __init__(self, t=None):
        self.id = uuid.uuid4()
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.tuples = set()
        self.bet_uniques_vectors = set()
        self.bet_uniques_words = set()
        if t is not None:
            self.tuples.add(t)

    def __eq__(self, other):
        return self.tuples == other.tuples

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence(self, config):
        if self.positive > 0:
            self.confidence = float(self.positive) / float(
                self.positive + self.unknown * config.wUnk + self.negative * config.wNeg
            )
        elif self.positive == 0:
            self.confidence = 0

    def add_tuple(self, t):
        self.tuples.add(t)

    def merge_all_tuples_bet(self) -> None:
        """Put all tuples with BET vectors into a set so that comparison with repeated vectors is eliminated"""
        self.bet_uniques_vectors = set()
        self.bet_uniques_words = set()
        for t in self.tuples:
            # transform numpy array into a tuple, so it can be hashed and added into a set
            self.bet_uniques_vectors.add(tuple(t.bet_vector))
            self.bet_uniques_words.add(t.bet_words)

    def update_selectivity(self, t, config):
        matched_both = False
        matched_e1 = False

        for s in config.positive_seed_tuples:
            if s.e1.strip() == t.e1.strip():
                matched_e1 = True
                if s.e2.strip() == t.e2.strip():
                    self.positive += 1
                    matched_both = True
                    break

        if matched_e1 is True and matched_both is False:
            self.negative += 1

        if matched_both is False:
            for n in config.negative_seed_tuples:
                if n.e1.strip() == t.e1.strip():
                    if n.e2.strip() == t.e2.strip():
                        self.negative += 1
                        matched_both = True
                        break

        if not matched_both and not matched_e1:
            self.unknown += 1
