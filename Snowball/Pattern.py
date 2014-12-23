__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Pattern(object):

    def __init__(self, t=None):
        self.positive = 0
        self.negative = 0
        self.confidence = 0
        self.tuples = set()
        self.centroid_bef = None
        self.centroid_bet = None
        self.centroid_aft = None
        if tuple is not None:
            self.tuples.add(t)
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def __str__(self):
        output = ''
        for t in self.tuples:
            output += str(t)+'\n'
        return output

    def update_confidence(self):
        if self.positive or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)
        self.centroid(self)

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
        self.update_confidence()

    @staticmethod
    def centroid(self):
        print "tuples in cluster", len(self.tuples)
        if len(self.tuples) == 1:
            t = next(iter(self.tuples))
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector
        else:
            print "Calculate centroid"
            # TODO: calculate this centroid"
            for t in self.tuples:
                print t

























